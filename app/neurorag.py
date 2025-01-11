import re
import llm_blender
from operator import itemgetter
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from NCBIProteinRetriever import NCBIProteinRetriever

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.llms import Ollama
from langgraph.graph import START, END, StateGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import PubMedRetriever, ArxivRetriever
from langchain_community.tools.tavily_search import TavilySearchResults

class RouteQuerySchema(BaseModel):
  sources: List[str] = Field(
    description='Given a user question select the retrieval methods you consider the most appropriate for addressing this question. You may also return an empty array if no methods are required.',
  )

class GradeDocumentsSchema(BaseModel):
  binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinationsSchema(BaseModel):
  binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswerSchema(BaseModel):
  binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

class DecompositionAnswerSchema(BaseModel):
  subqueries: List[str] = Field(description="Given the original query, decompose it into 2-4 simpler sub-queries as json array of strings")

class NCBIProteinDBAnswerSchema(BaseModel):
  query: str = Field(description='Given the original query, please find a protein locus for the NCBI protein database.')

class GraphStateSchema(TypedDict):
  question: str

  specialized_srcs: List[str]

  step_back_query: str
  rewritten_query: str
  subqueries: List[str]

  generated_docs: List[str]

  documents: Annotated[list, operator.add]

  web_search: str

  generation: str
  generations_num: int

class NeuroRAG():
  def __init__(self, temperature=0):
    self.temperature = temperature

  def compile(self):
    self.llm = Ollama(model='llama3.1', temperature=self.temperature)

    embeddings = OllamaEmbeddings(model='llama3.1')
    embeddings_store = LocalFileStore('./.embeddings_cache')
    self.embeddings = CacheBackedEmbeddings.from_bytes_store(
      embeddings,
      embeddings_store,
      namespace=embeddings.model,
    )

    self.vector_store = Chroma(
      collection_name='neurorag',
      embedding_function=self.embeddings,
      persist_directory='./chroma_db',
    )
    self.vector_store_retriever = self.vector_store.as_retriever()
    self.pub_med_retriever = PubMedRetriever()
    self.arxiv_retriever = ArxivRetriever(
      load_max_docs=3,
      get_ful_documents=True,
    )
    self.ncbi_protein_retriever = NCBIProteinRetriever(k=3)

    self.route_chain = self.__build_route_chain()
    self.docs_grader_chain = self.__build_docs_grader_chain()
    self.hallucinations_grader_chain = self.__build_hallucinations_grader_chain()
    self.answer_grader_chain = self.__build_answer_grader_chain()
    self.hyde_chain = self.__build_hyde_chain()
    self.step_back_chain = self.__build_step_back_chain()
    self.query_rewrite_chain = self.__build_query_rewrite_chain()
    self.decomposition_chain = self.__build_decomposition_chain()
    self.ncbi_protein_db_chain = self.__build_ncbi_protein_db_chain()
    self.rag_chain = self.__build_rag_chain()
    self.web_search_chain = self.__build_web_search_chain()

    workflow = StateGraph(GraphStateSchema)

    workflow.add_node(
      'determine_specialized_srcs',
      self.determine_specialized_src_node,
    )
    workflow.add_node(
      'generate_step_back_query',
      self.generate_step_back_query_node,
    )
    workflow.add_node(
      'generate_rewritten_query',
      self.generate_rewritten_query_node,
    )
    workflow.add_node(
      'generate_subqueries',
      self.generate_subqueries_node,
    )
    workflow.add_node('generate_hyde_docs', self.generate_hyde_docs_node)
    workflow.add_node(
      'vector_store_retriever',
      self.vector_store_retriever_node,
    )
    workflow.add_node('pub_med_retriever', self.pub_med_retriever_node)
    workflow.add_node('arxiv_retriever', self.arxiv_retriever_node)
    workflow.add_node('ncbi_protein_db_retriever', self.ncbi_protein_db_retriever_node)
    workflow.add_node('websearch', self.web_search_node)
    workflow.add_node('generate', self.generate_node)
    workflow.add_node('grade_documents', self.grade_documents_node)

    workflow.add_edge(START, 'determine_specialized_srcs')
    workflow.add_conditional_edges(
      'determine_specialized_srcs',
      self.route_question_node,
      {
        'websearch': 'websearch',
        'specialized_srcs': 'generate_step_back_query',
      },
    )
    workflow.add_edge('generate_step_back_query', 'generate_rewritten_query')
    workflow.add_edge('generate_rewritten_query', 'generate_subqueries')
    workflow.add_edge('generate_subqueries', 'generate_hyde_docs')
    workflow.add_edge('generate_hyde_docs', 'vector_store_retriever')
    workflow.add_edge('generate_hyde_docs', 'pub_med_retriever')
    workflow.add_edge('generate_hyde_docs', 'arxiv_retriever')
    workflow.add_edge('generate_hyde_docs', 'ncbi_protein_db_retriever')
    workflow.add_edge('vector_store_retriever', 'grade_documents')
    workflow.add_edge('pub_med_retriever', 'grade_documents')
    workflow.add_edge('arxiv_retriever', 'grade_documents')
    workflow.add_edge('ncbi_protein_db_retriever', 'grade_documents')
    workflow.add_conditional_edges(
      'grade_documents',
      self.decide_to_generate_node,
      {
        'websearch': 'websearch',
        'generate': 'generate',
      },
    )
    workflow.add_edge('websearch', 'generate')
    workflow.add_conditional_edges(
      'generate',
      self.grade_generation_node,
      {
        'not supported': 'generate',
        'useful': END,
        'not useful': 'websearch',
      },
    )

    self.app = workflow.compile()

  def invoke(self, question):
    result = self.app.invoke({'question': question})
    return result

  def extract_json_parser(self, response):
    json_pattern = r'\{.*?\}'
    match = re.search(json_pattern, response, re.DOTALL)

    if match:
      return match.group().strip()

    return response

  def __build_route_chain(self):
    parser = PydanticOutputParser(pydantic_object=RouteQuerySchema)
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=self.llm,
      max_retries=3,
    )
    template = """
You are an expert at selecting retrieval methods.
Given a user question select the retrieval methods you consider the most appropriate for addressing user question.
You may also return an empty array if no methods are required.

Possible retrieval methods:
1. The "vectorstore" retriever contains documents related to neurobiology and medicine. Use the vectorstore for questions on these topics.
2. The "pubmed" retriever contains biomedical literature and research articles. It is particularly useful for answering detailed questions about medical research, clinical studies, and scientific discoveries.
3. The "arxiv" retriever contains preprints of research papers across various scientific fields, including physics, mathematics, computer science, and biology. Use the arxiv for questions on recent scientific research and theoretical studies in these areas.
4. The "ncbi_protein" retriever contains protein sequence and functional information. Use the NCBI protein DB for questions related to protein sequences, structures, and functions.

{format_instructions}

User question:
{question}
    """
    prompt = PromptTemplate(
      template=template,
      input_variables=['question'],
      partial_variables={'format_instructions': parser.get_format_instructions()},
    )
    chain = RunnableParallel(
      completion=prompt | self.llm | self.extract_json_parser,
      prompt_value=prompt,
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    return chain

  def __build_docs_grader_chain(self):
    parser = PydanticOutputParser(pydantic_object=GradeDocumentsSchema)
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=self.llm,
      max_retries=3,
    )
    template = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

{format_instructions}

User question:
{question}

Retrieved document:
{document}
    """
    prompt = PromptTemplate(
      template=template,
      input_variables=['document', 'question'],
      partial_variables={'format_instructions': parser.get_format_instructions()},
    )
    chain = RunnableParallel(
      completion=prompt | self.llm | self.extract_json_parser,
      prompt_value=prompt,
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    return chain

  def __build_hallucinations_grader_chain(self):
    parser = PydanticOutputParser(pydantic_object=GradeHallucinationsSchema)
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=self.llm,
      max_retries=3,
    )
    template = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."

{format_instructions}

Set of facts:
{documents}

LLM generation:
{generation}
    """
    prompt = PromptTemplate(
      template=template,
      input_variables=['document', 'question'],
      partial_variables={'format_instructions': parser.get_format_instructions()},
    )
    chain = RunnableParallel(
      completion=prompt | self.llm | self.extract_json_parser,
      prompt_value=prompt,
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    return chain

  def __build_answer_grader_chain(self):
    parser = PydanticOutputParser(pydantic_object=GradeAnswerSchema)
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=self.llm,
      max_retries=3,
    )
    template = """
You are a grader assessing whether an answer addresses / resolves a question. \n
Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.

{format_instructions}

User question:
{question}

LLM generation:
{generation}
    """
    prompt = PromptTemplate(
      template=template,
      input_variables=['document', 'question'],
      partial_variables={'format_instructions': parser.get_format_instructions()},
    )
    chain = RunnableParallel(
      completion=prompt | self.llm | self.extract_json_parser,
      prompt_value=prompt,
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    return chain

  def __build_hyde_chain(self):
    template = """
Please write a scientific paper passage to answer the question

Question: {question}

Passage:
    """
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = prompt | self.llm | parser

    return chain

  def __build_step_back_chain(self):
    template = """
You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

Original query: {question}

Step-back query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = prompt | self.llm | parser

    return chain

  def __build_query_rewrite_chain(self):
    template = """
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {question}

Rewritten query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = prompt | self.llm | parser

    return chain

  def __build_decomposition_chain(self):
    parser = PydanticOutputParser(pydantic_object=DecompositionAnswerSchema)
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=self.llm,
      max_retries=3,
    )
    template = """
You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Original query: {question}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?

{format_instructions}
    """
    prompt = PromptTemplate(
      template=template,
      input_variables=['question'],
      partial_variables={'format_instructions': parser.get_format_instructions()},
    )
    chain = RunnableParallel(
      completion=prompt | self.llm | self.extract_json_parser,
      prompt_value=prompt,
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

    return chain

  def __build_ncbi_protein_db_chain(self):
    parser = PydanticOutputParser(pydantic_object=NCBIProteinDBAnswerSchema)
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=self.llm,
      max_retries=3,
    )

    template = """
As an expert in bioinformatics and user query optimization for biological databases, your task is to transform user questions into precise and effective queries suitable for the NCBI protein database.
Create a query with only locus of a protein for search within the NCBI protein database.

Original query: {question}

{format_instructions}
    """
    prompt = PromptTemplate(
      template=template,
      input_variables=['question'],
      partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    query_extractor = lambda res: res.query

    chain = RunnableParallel(
      completion=prompt | self.llm | self.extract_json_parser,
      prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x)) | query_extractor | self.ncbi_protein_retriever

    return chain

  def __build_rag_chain(self):
    blender = llm_blender.Blender()
    blender.loadranker('llm-blender/PairRM', device='mps')
    blender.loadfuser('llm-blender/gen_fuser_3b', device='mps')

    prompt = hub.pull('rlm/rag-prompt')

    gpt_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    openbio_llm = Ollama(model='taozhiyuai/openbiollm-llama-3:70b_q2_k', temperature=0)
    biomistral_llm = Ollama(model='cniongolo/biomistral', temperature=0)

    gpt_chain = prompt | gpt_llm | StrOutputParser()
    openbio_chain = prompt | openbio_llm | StrOutputParser()
    biomistral_chain = prompt | biomistral_llm | StrOutputParser()

    def fuse_generations(dict):
      question = dict['question']

      gpt_res = dict['gpt_res']
      openbio_res = dict['openbio_res']
      biomistral_res = dict['biomistral_res']
      answers = [gpt_res, openbio_res, biomistral_res]

      fuse_generations, ranks = blender.rank_and_fuse(
        [question],
        [answers],
        instructions=[''],
        return_scores=False,
        batch_size=2,
        top_k=3
      )
      return fuse_generations[0]

    chain = (
      {
        'llama_res': llama_chain,
        'mistral_res': mistral_chain,
        'gpt_res': gpt_chain,
        'question': itemgetter('question')
      }
      | RunnableLambda(fuse_generations)
    )

    return chain

  def __build_web_search_chain(self):
    chain = TavilySearchResults(k=5)

    return chain

  def determine_specialized_src_node(self, state):
    print('---DETERMINE SPECIALIZED SOURCES---')

    question = state['question']

    try:
      res = self.route_chain.invoke({'question': question})
      srcs = [src.strip().lower() for src in res.sources]
    except:
      srcs = []

    return {'specialized_srcs': srcs}

  def route_question_node(self, state):
    print('---ROUTE QUESTION---')

    sources = state['specialized_srcs']

    if len(sources) == 0:
      print('---ROUTE QUESTION TO WEB SEARCH---')
      return 'websearch'
    else:
      print(f'---ROUTE QUESTION TO SPECIALIZED SOURCES: {", ".join([source.upper() for source in sources])}---')
      return 'specialized_srcs'

  def generate_step_back_query_node(self, state):
    print('---GENERATE STEP-BACK QUERY---')

    question = state['question']
    step_back_query = self.step_back_chain.invoke({'question': question})
    return {'step_back_query': step_back_query}

  def generate_rewritten_query_node(self, state):
    print('---GENERATE REWRITTEN QUERY---')

    question = state['question']
    rewritten_query = self.query_rewrite_chain.invoke({'question': question})
    return {'rewritten_query': rewritten_query}

  def generate_subqueries_node(self, state):
    print('---GENERATE SUBQUERIES---')

    question = state['question']

    try:
      decomposition_answer = self.decomposition_chain.invoke({'question': question})
      subqueries = decomposition_answer.subqueries
      # Limit to a maximum of four subqueries
      subqueries = subqueries[:4]
    except:
      subqueries = []

    print(f'---FINAL SUBQUERIES NUMBER: {len(subqueries)}---')

    return {'subqueries': subqueries}

  def generate_hyde_docs_node(self, state):
    print('---GENERATE HYDE DOCUMENTS---')

    question = state['question']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [question, step_back_query, rewritten_query, *subqueries]
    generated_docs = []

    for query in queries:
      generated_doc = self.hyde_chain.invoke({'question': query})
      generated_docs.append(generated_doc)

    return {'question': question, 'generated_docs': generated_docs}

  def vector_store_retriever_node(self, state):
    generated_docs = state['generated_docs']
    specialized_srcs = state['specialized_srcs']

    if 'vectorstore' not in specialized_srcs:
      return {'documents': []}

    print('---RETRIEVE FROM VECTOR STORE---')

    documents = []

    for generated_doc in generated_docs:
      documents.extend(self.vector_store_retriever.invoke(generated_doc))

    return {'documents': documents}

  def pub_med_retriever_node(self, state):
    generated_docs = state['generated_docs']
    specialized_srcs = state['specialized_srcs']

    if 'pubmed' not in specialized_srcs:
      return {'documents': []}

    print('---RETRIEVE FROM PUBMED---')

    documents = []

    for generated_doc in generated_docs:
      try:
        documents.extend(self.pub_med_retriever.invoke(generated_doc))
      except:
        pass

    return {'documents': documents}

  def arxiv_retriever_node(self, state):
    generated_docs = state['generated_docs']
    specialized_srcs = state['specialized_srcs']

    if 'arxiv' not in specialized_srcs:
      return {'documents': []}

    print('---RETRIEVE FROM ARXIV---')

    documents = []

    for generated_doc in generated_docs:
      try:
        documents.extend(self.arxiv_retriever.invoke(generated_doc))
      except:
        pass

    return {'documents': documents}

  def ncbi_protein_db_retriever_node(self, state):
    specialized_srcs = state['specialized_srcs']

    if 'ncbi_protein' not in specialized_srcs:
      return {'documents': []}

    print('---RETRIEVE FROM NCBI PROTEIN DB---')

    question = state['question']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [question, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.ncbi_protein_db_chain.invoke(query))
      except:
        pass

    return {'documents': documents}

  def grade_documents_node(self, state):
    print('---CHECK DOCUMENT RELEVANCE TO QUESTION---')

    question = state['question']
    documents = state['documents']

    print(f'---INITIAL DOCUMENTS NUMBER: {len(documents)}---')

    filtered_documents = []
    seen_contents = set()
    web_search = 'No'

    for index, document in enumerate(documents):
      print(f'---GRADE DOCUMENT ({index + 1}/{len(documents)})---')

      if document.page_content in seen_contents:
        print('---GRADE: DOCUMENT IS REPEATED---')
        continue
      seen_contents.add(document.page_content)

      try:
        score = self.docs_grader_chain.invoke({
          'question': question,
          'document': document.page_content,
        })
        grade = score.binary_score
      except:
        grade = 'no'

      if grade.lower() == 'yes':
        print('---GRADE: DOCUMENT RELEVANT---')
        filtered_documents.append(document)
      else:
        print('---GRADE: DOCUMENT NOT RELEVANT---')
        web_search = 'Yes'
        continue

    print(f'---FINAL DOCUMENTS NUMBER: {len(filtered_documents)}---')

    state['documents'].clear()
    return {
      'documents': filtered_documents,
      'web_search': web_search,
    }

  def decide_to_generate_node(self, state):
    web_search = state['web_search']
    return 'websearch' if web_search == 'Yes' else 'generate'

  def web_search_node(self, state):
    print('---WEB SEARCH---')

    question = state['question']

    web_results = self.web_search_chain.invoke({'query': question})
    docs = [Document(page_content=result['content'], metadata={'source': result['url']}) for result in web_results]

    return {'documents': docs}

  def generate_node(self, state):
    print('---GENERATE---')

    question = state['question']
    documents = state['documents']
    generations_num = state.get('generations_num', 0)

    context = '\n\n' + '\n\n'.join(map(lambda doc: doc.page_content, documents)) + '\n\n'
    generation = self.rag_chain.invoke({'context': context, 'question': question})

    return {'generation': generation, 'generations_num': generations_num + 1}

  def grade_generation_node(self, state):
    print('---CHECK HALLUCINATIONS---')

    question = state['question']
    documents = state['documents']
    generation = state['generation']
    generations_num = state['generations_num']

    if generations_num >= 2:
      return 'useful'

    try:
      context = '\n\n' + '\n\n'.join(map(lambda doc: doc.page_content, documents)) + '\n\n'
      score = self.hallucinations_grader_chain.invoke({
        'documents': context,
        'generation': generation,
      })
      grade = score.binary_score
    except:
      grade = 'no'

    if grade == 'yes':
      print('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---')
      print('---GRADE GENERATION vs QUESTION---')

      try:
        score = self.answer_grader_chain.invoke({
          'question': question,
          'generation': generation,
        })
        grade = score.binary_score.lower()
      except:
        grade = 'no'

      if grade == 'yes':
        print('---DECISION: GENERATION ADDRESSES QUESTION---')
        return 'useful'
      else:
        print('---DECISION: GENERATION DOES NOT ADDRESS QUESTION---')
        return 'not useful'
    else:
      print('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---')
      return 'not supported'
