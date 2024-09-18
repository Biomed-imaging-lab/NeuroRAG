import os
import re

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.llms import Ollama
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document

class RouteQuerySchema(BaseModel):
  data_source: Literal['vectorstore', 'websearch'] = Field(
    description='Given a user question choose to route it to web search or a vectorstore.',
  )

class GradeDocumentsSchema(BaseModel):
  binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinationsSchema(BaseModel):
  binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswerSchema(BaseModel):
  binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

class GraphStateSchema(TypedDict):
  question: str
  generated_doc: str
  documents: List[str]
  web_search: str
  generation: str
  generations_num: int
  sources_used: List[str]

class BioRAG():
  def __init__(self, docs=[], temperature=0):
    self.docs = docs
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
      embedding_function=self.embeddings,
      persist_directory='./chroma_db',
    )
    self.retriever = self.vector_store.as_retriever()

    self.route_chain = self.__build_route_chain()
    self.docs_grader_chain = self.__build_docs_grader_chain()
    self.hallucinations_grader_chain = self.__build_hallucinations_grader_chain()
    self.answer_grader_chain = self.__build_answer_grader_chain()
    self.hyde_chain = self.__build_hyde_chain()
    self.rag_chain = self.__build_rag_chain()
    self.web_search_chain = self.__build_web_search_chain()

    workflow = StateGraph(GraphStateSchema)
    workflow.add_node('generate_doc', self.generate_doc_node)
    workflow.add_node('retrieve', self.retrieve_node)
    workflow.add_node('websearch', self.web_search_node)
    workflow.add_node('generate', self.generate_node)
    workflow.add_node('grade_documents', self.grade_documents_node)
    workflow.set_conditional_entry_point(
      self.route_question_node,
      {
        'websearch': 'websearch',
        'vectorstore': 'generate_doc',
      },
    )
    workflow.add_edge('generate_doc', 'retrieve')
    workflow.add_edge('retrieve', 'grade_documents')
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
    return result['generation']

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
    You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to neurobiology and medicine.
    Use the vectorstore for questions on these topics. For all else, use web-search.

    {format_instructions}

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

  def __build_rag_chain(self):
    prompt = hub.pull('rlm/rag-prompt')
    parser = StrOutputParser()
    chain = prompt | self.llm | parser

    return chain

  def __build_web_search_chain(self):
    os.environ['TAVILY_API_KEY'] = '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'
    chain = TavilySearchResults(k=5)

    return chain

  def route_question_node(self, state):
    question = state['question']
    source = self.route_chain.invoke({'question': question})
    return source.data_source

  def generate_doc_node(self, state):
    question = state['question']
    generated_doc = self.hyde_chain.invoke({'question': question})
    return {'question': question, 'generated_doc': generated_doc}

  def retrieve_node(self, state):
    question = state['question']
    generated_doc = state['generated_doc']
    print('generated_doc', generated_doc)
    docs = self.retriever.invoke(generated_doc)
    print('vs docs', docs)
    return {'question': question, 'documents': docs}

  def grade_documents_node(self, state):
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    web_search = 'No'

    for doc in documents:
      try:
        score = self.docs_grader_chain.invoke({'question': question, 'document': doc.page_content})
        grade = score.binary_score.lower()
      except:
        grade = 'no'

      if grade == 'yes':
        filtered_docs.append(doc)
      else:
        web_search = 'Yes'
        continue

    return {
      'question': question,
      'documents': filtered_docs,
      'web_search': web_search,
    }

  def decide_to_generate_node(self, state):
    web_search = state['web_search']
    return 'websearch' if web_search == 'Yes' else 'generate'

  def web_search_node(self, state):
    question = state['question']
    documents = state.get('documents')

    try:
      docs = self.web_search_chain.invoke({'query': question})
      web_results = '\n'.join([d['content'] for d in docs])
      web_results = Document(page_content=web_results)

      if documents is not None:
        documents.append(web_results)
      else:
        documents = [web_results]
    except:
      pass

    return {
      'question': question,
      'documents': documents,
    }

  def generate_node(self, state):
    question = state['question']
    documents = state['documents']
    generations_num = state.get('generations_num', 0) or 0

    generation = self.rag_chain.invoke({'context': documents, 'question': question})

    return {
      'question': question,
      'documents': documents,
      'generation': generation,
      'generations_num': generations_num + 1,
    }

  def grade_generation_node(self, state):
    question = state['question']
    documents = state['documents']
    generation = state['generation']
    generations_num = state['generations_num']

    if generations_num >= 2:
      return 'useful'

    try:
      score = self.hallucinations_grader_chain.invoke({
        'documents': documents,
        'generation': generation,
      })
      grade = score.binary_score
    except:
      grade = 'no'

    if grade == 'yes':
      try:
        score = self.answer_grader_chain.invoke({
          'question': question,
          'generation': generation,
        })
        grade = score.binary_score.lower()
      except:
        grade = 'no'

      return 'useful' if grade == 'yes' else 'not useful'
    else:
      return 'not supported'


