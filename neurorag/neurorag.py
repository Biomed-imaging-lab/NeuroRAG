import re
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.llms import Ollama
from langgraph.graph import START, END, StateGraph
from langchain_community.retrievers import PubMedRetriever, ArxivRetriever, BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults

from chains.route import RouteChain
from chains.document_grade import DocumentGradeChain
from chains.hallucinations import HallucinationsChain
from chains.answer_grade import AnswerGradeChain
from chains.hyde import HyDEChain
from chains.step_back import StepBackChain
from chains.query_rewriting import QueryRewritingChain
from chains.decomposition import DecompositionChain
from chains.ncbi_protein import NCBIProteinChain
from chains.ncbi_gene import NCBIGeneChain
from chains.generation import GenerationChain

class GraphStateSchema(TypedDict):
  question: str

  specialized_sources: list[str]

  step_back_query: str
  rewritten_query: str
  subqueries: list[str]

  generated_documents: list[str]

  documents: Annotated[list, operator.add]

  web_search: bool

  generation: str
  generations_number: int

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

    self.route_chain = RouteChain(self.llm)
    self.hyde_chain = HyDEChain(self.llm)
    self.step_back_chain = StepBackChain(self.llm)
    self.query_rewrite_chain = QueryRewritingChain(self.llm)
    self.decomposition_chain = DecompositionChain(self.llm)
    self.ncbi_protein_db_chain = NCBIProteinChain(self.llm)
    self.ncbi_gene_db_chain = NCBIGeneChain(self.llm)
    self.document_grade_chain = DocumentGradeChain(self.llm)
    self.web_search_chain = TavilySearchResults(k=5)
    self.generation_chain = GenerationChain()
    self.hallucinations_chain = HallucinationsChain(self.llm)
    self.answer_grade_chain = AnswerGradeChain(self.llm)

    workflow = StateGraph(GraphStateSchema)

    workflow.add_node('determine_specialized_sources', self.determine_specialized_src_node)

    workflow.add_node('generate_step_back_query', self.generate_step_back_query_node)
    workflow.add_node('generate_rewritten_query', self.generate_rewritten_query_node)
    workflow.add_node('generate_subqueries', self.generate_subqueries_node)

    workflow.add_node('generate_hyde_documents', self.generate_hyde_documents_node)

    workflow.add_node('vector_store_retriever', self.vector_store_retriever_node)
    workflow.add_node('pub_med_retriever', self.pub_med_retriever_node)
    workflow.add_node('arxiv_retriever', self.arxiv_retriever_node)
    workflow.add_node('ncbi_protein_db_retriever', self.ncbi_protein_db_retriever_node)
    workflow.add_node('ncbi_gene_db_retriever', self.ncbi_gene_db_retriever_node)

    workflow.add_node('websearch', self.web_search_node)
    workflow.add_node('generate', self.generate_node)
    workflow.add_node('grade_documents', self.grade_documents_node)

    workflow.add_edge(START, 'determine_specialized_sources')
    workflow.add_conditional_edges(
      'determine_specialized_sources',
      self.route_question_node,
      {
        'websearch': 'websearch',
        'specialized_sources': 'generate_step_back_query',
      },
    )

    workflow.add_edge('generate_step_back_query', 'generate_rewritten_query')
    workflow.add_edge('generate_rewritten_query', 'generate_subqueries')
    workflow.add_edge('generate_subqueries', 'generate_hyde_documents')

    workflow.add_edge('generate_hyde_documents', 'vector_store_retriever')
    workflow.add_edge('generate_hyde_documents', 'pub_med_retriever')
    workflow.add_edge('generate_hyde_documents', 'arxiv_retriever')
    workflow.add_edge('generate_hyde_documents', 'ncbi_protein_db_retriever')
    workflow.add_edge('generate_hyde_documents', 'ncbi_gene_db_retriever')

    workflow.add_edge('vector_store_retriever', 'grade_documents')
    workflow.add_edge('pub_med_retriever', 'grade_documents')
    workflow.add_edge('arxiv_retriever', 'grade_documents')
    workflow.add_edge('ncbi_protein_db_retriever', 'grade_documents')
    workflow.add_edge('ncbi_gene_db_retriever', 'grade_documents')

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

  def route_question_node(self, state):
    sources = state['specialized_sources']
    return 'websearch' if len(sources) == 0 else 'specialized_sources'

  def generate_step_back_query_node(self, state):
    question = state['question']
    step_back_query = self.step_back_chain.invoke({'question': question})
    return {'step_back_query': step_back_query}

  def generate_rewritten_query_node(self, state):
    question = state['question']
    rewritten_query = self.query_rewrite_chain.invoke({'question': question})
    return {'rewritten_query': rewritten_query}

  def generate_subqueries_node(self, state):
    question = state['question']

    try:
      decomposition_answer = self.decomposition_chain.invoke({'question': question})
      subqueries = decomposition_answer.subqueries
      # Limit to a maximum of four subqueries
      subqueries = subqueries[:4]
    except:
      subqueries = []

    return {'subqueries': subqueries}

  def generate_hyde_documents_node(self, state):
    question = state['question']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [question, step_back_query, rewritten_query, *subqueries]
    generated_documents = []

    for query in queries:
      generated_document = self.hyde_chain.invoke({'question': query})
      generated_documents.append(generated_document)

    return {'question': question, 'generated_documents': generated_documents}

  def vector_store_retriever_node(self, state):
    generated_documents = state['generated_documents']
    specialized_sources = state['specialized_sources']

    if 'vectorstore' not in specialized_sources:
      return {'documents': []}

    documents = []

    for generated_document in generated_documents:
      documents.extend(self.vector_store_retriever.invoke(generated_document))

    return {'documents': documents}

  def pub_med_retriever_node(self, state):
    generated_documents = state['generated_documents']
    specialized_sources = state['specialized_sources']

    if 'pubmed' not in specialized_sources:
      return {'documents': []}

    documents = []

    for generated_document in generated_documents:
      try:
        documents.extend(self.pub_med_retriever.invoke(generated_document))
      except:
        pass

    return {'documents': documents}

  def arxiv_retriever_node(self, state):
    generated_documents = state['generated_documents']
    specialized_sources = state['specialized_sources']

    if 'arxiv' not in specialized_sources:
      return {'documents': []}

    documents = []

    for generated_document in generated_documents:
      try:
        documents.extend(self.arxiv_retriever.invoke(generated_document))
      except:
        pass

    return {'documents': documents}

  def ncbi_protein_db_retriever_node(self, state):
    specialized_sources = state['specialized_sources']

    if 'ncbi_protein' not in specialized_sources:
      return {'documents': []}

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

  def ncbi_gene_db_retriever_node(self, state):
    specialized_sources = state['specialized_sources']

    if 'ncbi_gene' not in specialized_sources:
      return {'documents': []}

    question = state['question']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [question, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.ncbi_gene_db_chain.invoke(query))
      except:
        pass

    return {'documents': documents}

  def grade_documents_node(self, state):
    rewritten_query = state['rewritten_query']
    documents = state['documents']

    if len(documents) == 0:
      return {'documents': [], 'web_search': True}

    unique_documents = list({doc.page_content: doc for doc in documents}.values())
    retriever = BM25Retriever.from_documents(unique_documents)
    retrieved_documents = retriever.invoke(rewritten_query)
    filtered_documents = []
    web_search = False

    for index, document in enumerate(retrieved_documents):
      try:
        score = self.document_grade_chain.invoke({
          'question': rewritten_query,
          'document': document.page_content,
        })
        grade = score.binary_score
      except:
        grade = 'No'

      if grade.lower() == 'yes':
        filtered_documents.append(document)
      else:
        web_search = True

    state['documents'].clear()
    return {
      'documents': filtered_documents,
      'web_search': len(filtered_documents) == 0,
    }

  def decide_to_generate_node(self, state):
    web_search = state['web_search']
    return 'websearch' if web_search else 'generate'

  def web_search_node(self, state):
    question = state['question']

    web_results = self.web_search_chain.invoke({'query': question})
    documents = [Document(page_content=result['content'], metadata={'source': result['url']}) for result in web_results]

    return {'documents': documents}

  def generate_node(self, state):
    question = state['question']
    documents = state['documents']
    generations_number = state.get('generations_number', 0)

    context = '\n\n' + '\n\n'.join(map(lambda doc: doc.page_content, documents)) + '\n\n'
    generation = self.generation_chain.invoke({'context': context, 'question': question})

    return {'generation': generation, 'generations_number': generations_number + 1}

  def grade_generation_node(self, state):
    question = state['question']
    documents = state['documents']
    generation = state['generation']
    generations_number = state['generations_number']

    if generations_number >= 2:
      return 'useful'

    try:
      context = '\n\n' + '\n\n'.join(map(lambda doc: doc.page_content, documents)) + '\n\n'
      score = self.hallucinations_chain.invoke({
        'documents': context,
        'generation': generation,
      })
      grade = score.binary_score
    except:
      grade = 'no'

    if grade == 'yes':
      try:
        score = self.answer_grade_chain.invoke({
          'question': question,
          'generation': generation,
        })
        grade = score.binary_score.lower()
      except:
        grade = 'no'

      if grade == 'yes':
        return 'useful'
      else:
        return 'not useful'
    else:
      return 'not supported'
