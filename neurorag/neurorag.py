import operator
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_ollama.llms import OllamaLLM as Ollama
from langgraph.graph import START, END, StateGraph
from langchain_community.retrievers import (
  PubMedRetriever,
  ArxivRetriever,
  BM25Retriever,
)
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
from chains.biorxiv import BioRxivChain, MedRxivChain
from chains.generation import GenerationChain
from models.OpenRouter import OpenRouter


class GraphStateSchema(TypedDict):
  query: str

  specialized_sources: list[str]

  step_back_query: str
  rewritten_query: str
  subqueries: list[str]

  generated_documents: list[str]

  documents: Annotated[list, operator.add]

  web_search: bool
  web_results: list[Document]

  generation: str
  generations_number: int


ollama_server_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')


class NeuroRAG:
  def __init__(
    self,
    model='meta-llama/llama-3.3-70b-instruct',
    temperature: float = 0,
    debug: bool = False,
    generation_prompt=None,
    max_retries: int = 2,
    llms=None,
  ) -> None:
    self.temperature = temperature
    self.debug = debug
    self.generation_prompt = generation_prompt
    self.max_retries = max_retries
    self.llms = llms
    self.llm = OpenRouter(model=model, temperature=self.temperature)

  def compile(self) -> None:
    embeddings = OllamaEmbeddings(model='llama3.1', base_url=ollama_server_url)
    embeddings_store = LocalFileStore('./.embeddings_cache')
    self.embeddings = CacheBackedEmbeddings.from_bytes_store(
      embeddings,
      embeddings_store,
      namespace=embeddings.model,
    )

    self.vector_store = Chroma(
      collection_name='neurorag',
      embedding_function=self.embeddings,
      persist_directory='../chroma_db',
    )
    self.vector_store_retriever = self.vector_store.as_retriever()
    self.pub_med_retriever = PubMedRetriever(top_k_results=5)
    self.arxiv_retriever = ArxivRetriever(load_max_docs=3, get_ful_documents=True)

    self.route_chain = RouteChain(self.llm)
    self.hyde_chain = HyDEChain(self.llm)
    self.step_back_chain = StepBackChain(self.llm)
    self.query_rewrite_chain = QueryRewritingChain(self.llm)
    self.decomposition_chain = DecompositionChain(self.llm)
    self.ncbi_protein_db_chain = NCBIProteinChain(self.llm)
    self.ncbi_gene_db_chain = NCBIGeneChain(self.llm)
    self.biorxiv_chain = BioRxivChain(self.llm)
    self.medrxiv_chain = MedRxivChain(self.llm)
    self.document_grade_chain = DocumentGradeChain(self.llm)
    self.web_search_chain = TavilySearchResults(k=self.max_retries * 3)
    self.generation_chain = GenerationChain(self.llm, self.temperature, llms=self.llms)
    self.hallucinations_chain = HallucinationsChain(self.llm)
    self.answer_grade_chain = AnswerGradeChain(self.llm)

    workflow = StateGraph(GraphStateSchema)

    workflow.add_node(
      'determine_specialized_sources', self.determine_specialized_src_node
    )

    workflow.add_node('generate_step_back_query', self.generate_step_back_query_node)
    workflow.add_node('generate_rewritten_query', self.generate_rewritten_query_node)
    workflow.add_node('generate_subqueries', self.generate_subqueries_node)

    workflow.add_node('generate_hyde_documents', self.generate_hyde_documents_node)

    workflow.add_node('vector_store_retriever', self.vector_store_retriever_node)
    workflow.add_node('pub_med_retriever', self.pub_med_retriever_node)
    workflow.add_node('arxiv_retriever', self.arxiv_retriever_node)
    workflow.add_node('ncbi_protein_db_retriever', self.ncbi_protein_db_retriever_node)
    workflow.add_node('ncbi_gene_db_retriever', self.ncbi_gene_db_retriever_node)
    workflow.add_node('biorxiv_retriever', self.biorxiv_retriever_node)
    workflow.add_node('medrxiv_retriever', self.medrxiv_retriever_node)

    workflow.add_node('websearch', self.web_search_node)
    workflow.add_node('generate', self.generate_node)
    workflow.add_node('grade_documents', self.grade_documents_node)

    workflow.add_edge(START, 'determine_specialized_sources')
    workflow.add_conditional_edges(
      'determine_specialized_sources',
      self.route_query_node,
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
    workflow.add_edge('generate_hyde_documents', 'biorxiv_retriever')
    workflow.add_edge('generate_hyde_documents', 'medrxiv_retriever')

    workflow.add_edge('vector_store_retriever', 'grade_documents')
    workflow.add_edge('pub_med_retriever', 'grade_documents')
    workflow.add_edge('arxiv_retriever', 'grade_documents')
    workflow.add_edge('ncbi_protein_db_retriever', 'grade_documents')
    workflow.add_edge('ncbi_gene_db_retriever', 'grade_documents')
    workflow.add_edge('biorxiv_retriever', 'grade_documents')
    workflow.add_edge('medrxiv_retriever', 'grade_documents')

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

  def invoke(self, query: str):
    result = self.app.invoke({'query': query})
    return result

  def determine_specialized_src_node(self, state):
    query = state['query']

    if self.debug:
      print('---DETERMINE SPECIALIZED SOURCES---')

    try:
      sources = self.route_chain.invoke(query)
      specialized_sources = [source.strip().lower() for source in sources]
    except:
      specialized_sources = []

    if self.debug:
      print(f'---SELECTED SOURCES: {specialized_sources}---')

    return {'specialized_sources': specialized_sources}

  def route_query_node(
    self, state: GraphStateSchema
  ) -> Literal['websearch', 'specialized_sources']:
    sources = state['specialized_sources']

    if self.debug:
      print('---ROUTE QUESTION---')

    return 'websearch' if len(sources) == 0 else 'specialized_sources'

  def generate_step_back_query_node(self, state: GraphStateSchema):
    query = state['query']

    if self.debug:
      print('---GENERATE STEP-BACK QUERY---')

    try:
      step_back_query = self.step_back_chain.invoke(query)
    except Exception as e:
      if self.debug:
        print('generate_step_back_query_node', e)
      step_back_query = query

    return {'step_back_query': step_back_query}

  def generate_rewritten_query_node(self, state: GraphStateSchema):
    query = state['query']

    if self.debug:
      print('---GENERATE REWRITTEN QUERY---')

    try:
      rewritten_query = self.query_rewrite_chain.invoke(query)
    except Exception as e:
      if self.debug:
        print('generate_step_back_query_node', e)
      rewritten_query = query

    return {'rewritten_query': rewritten_query}

  def generate_subqueries_node(self, state: GraphStateSchema):
    query = state['query']

    if self.debug:
      print('---GENERATE SUBQUERIES---')

    try:
      subqueries = self.decomposition_chain.invoke(query)
      # Limit to a maximum of four subqueries
      subqueries = subqueries[:4]
    except Exception as e:
      if self.debug:
        print('generate_subqueries_node', e)
      subqueries = []

    return {'subqueries': subqueries}

  def generate_hyde_documents_node(self, state: GraphStateSchema):
    query = state['query']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    if self.debug:
      print('---GENERATE HYDE DOCUMENTS---')

    queries = [query, step_back_query, rewritten_query, *subqueries]
    generated_documents = []

    for query in queries:
      generated_document = self.hyde_chain.invoke(query)
      generated_documents.append(generated_document)

    return {'generated_documents': generated_documents}

  def vector_store_retriever_node(self, state: GraphStateSchema):
    generated_documents = state['generated_documents']
    specialized_sources = state['specialized_sources']

    if 'vectorstore' not in specialized_sources:
      return {'documents': []}

    if self.debug:
      print('---RETRIEVE FROM VECTOR STORE---')

    documents = []

    for generated_document in generated_documents:
      documents.extend(self.vector_store_retriever.invoke(generated_document))

    return {'documents': documents}

  def pub_med_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']
    query = state['query']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    if 'pubmed' not in specialized_sources:
      return {'documents': []}

    if self.debug:
      print('---RETRIEVE FROM PUBMED---')

    queries = [query, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.pub_med_retriever.invoke(query))
      except Exception as e:
        if self.debug:
          print('pub_med_retriever_node', e)

    for document in documents:
      document.metadata['source'] = document.metadata['Title']

    return {'documents': documents}

  def arxiv_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']
    query = state['query']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    if 'arxiv' not in specialized_sources:
      return {'documents': []}

    if self.debug:
      print('---RETRIEVE FROM ARXIV---')

    queries = [query, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.arxiv_retriever.invoke(query))
      except Exception as e:
        if self.debug:
          print('arxiv_retriever_node', e)

    for document in documents:
      document.metadata['source'] = document.metadata['Title']

    return {'documents': documents}

  def ncbi_protein_db_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'ncbi_protein' not in specialized_sources:
      return {'documents': []}

    if self.debug:
      print('---RETRIEVE FROM NCBI PROTEIN DB---')

    query = state['query']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [query, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.ncbi_protein_db_chain.invoke(query))
      except Exception as e:
        if self.debug:
          print('ncbi_protein_db_retriever_node', e)
        pass

    return {'documents': documents}

  def ncbi_gene_db_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'ncbi_gene' not in specialized_sources:
      return {'documents': []}

    if self.debug:
      print('---RETRIEVE FROM NCBI GENE DB---')

    query = state['query']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [query, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.ncbi_gene_db_chain.invoke(query))
      except Exception as e:
        if self.debug:
          print('ncbi_gene_db_retriever_node', e)

    return {'documents': documents}

  def biorxiv_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'biorxiv' not in specialized_sources:
      return {'documents': []}

    if self.debug:
      print('---RETRIEVE FROM BIOREVIV---')

    query = state['query']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [query, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.biorxiv_chain.invoke(query))
      except Exception as e:
        if self.debug:
          print('biorxiv_retriever_node', e)

    return {'documents': documents}

  def medrxiv_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'medrxiv' not in specialized_sources:
      return {'documents': []}

    if self.debug:
      print('---RETRIEVE FROM MEDRXIV---')

    query = state['query']
    step_back_query = state['step_back_query']
    rewritten_query = state['rewritten_query']
    subqueries = state['subqueries']

    queries = [query, step_back_query, rewritten_query, *subqueries]
    documents = []

    for query in queries:
      try:
        documents.extend(self.medrxiv_chain.invoke(query))
      except Exception as e:
        if self.debug:
          print('medrxiv_retriever_node', e)

    return {'documents': documents}

  def grade_documents_node(self, state: GraphStateSchema):
    rewritten_query = state['rewritten_query']
    documents = state['documents']

    if self.debug:
      print('---GRADE DOCUMENTs---')

    if len(documents) == 0:
      return {'documents': [], 'web_search': True}

    unique_documents = list({doc.page_content: doc for doc in documents}.values())

    if len(unique_documents) > 10:
      retriever = BM25Retriever.from_documents(unique_documents)
      retrieved_documents = retriever.invoke(rewritten_query)
    else:
      retrieved_documents = unique_documents

    filtered_documents = []

    for document in retrieved_documents:
      try:
        grade = self.document_grade_chain.invoke(rewritten_query, document)
      except Exception as e:
        if self.debug:
          print('grade_documents_node', e)
        grade = 'no'

      if grade.lower() == 'yes':
        filtered_documents.append(document)

    filtered_documents = filtered_documents[:10]

    if self.debug:
      print(f'---FINAL DOCUMENTS NUMBER: {len(filtered_documents)}---')

    state['documents'].clear()
    return {
      'documents': filtered_documents,
      'web_search': len(filtered_documents) < 3,
    }

  def decide_to_generate_node(self, state: GraphStateSchema):
    web_search = state['web_search']

    if self.debug:
      print('---ASSESS GRADED DOCUMENTS---')

    if web_search:
      if self.debug:
        print(
          '---DECISION: SOME DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---'
        )
      return 'websearch'
    else:
      if self.debug:
        print('---DECISION: GENERATE---')
      return 'generate'

  def web_search_node(self, state: GraphStateSchema):
    query: str = state['query']
    web_results: list[Document] = state.get('web_results', [])
    generations_number: int = state.get('generations_number', 0)

    if self.debug:
      print('---WEB SEARCH---')

    if not web_results:
      try:
        raw_web_results = self.web_search_chain.invoke(query)
        web_results = [
          Document(page_content=result['content'], metadata={'source': result['url']})
          for result in raw_web_results
        ]
      except Exception as e:
        if self.debug:
          print('web_search_node', e)
        web_results = []

    new_documents = web_results[generations_number * 3 : generations_number * 3 + 3]

    return {'documents': new_documents, 'web_results': web_results}

  def generate_node(self, state: GraphStateSchema):
    query = state['query']
    documents = state['documents']
    generations_number = state.get('generations_number', 0)

    if self.debug:
      print('---GENERATE---')

    context = '\n\n'.join(map(lambda doc: doc.page_content, documents))
    generation = self.generation_chain.invoke(query, context, self.generation_prompt)

    return {'generation': generation, 'generations_number': generations_number + 1}

  def grade_generation_node(
    self, state: GraphStateSchema
  ) -> Literal['useful', 'not useful', 'not supported']:
    query = state['query']
    documents = state['documents']
    generation = state['generation']
    generations_number = state['generations_number']

    if self.debug:
      print('---GRADE GENERATION---')

    if generations_number >= self.max_retries:
      return 'useful'

    try:
      context = (
        '\n\n' + '\n\n'.join(map(lambda doc: doc.page_content, documents)) + '\n\n'
      )
      grade = self.hallucinations_chain.invoke(generation, context)
    except Exception as e:
      if self.debug:
        print('grade_generation_node hallucinations', e)
      grade = 'no'

    if grade == 'yes':
      if self.debug:
        print('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---')
      try:
        grade = self.answer_grade_chain.invoke(query, generation)
      except Exception as e:
        if self.debug:
          print('grade_generation_node answer_grade', e)
        grade = 'no'

      if grade == 'yes':
        if self.debug:
          print('---DECISION: GENERATION ADDRESSES QUESTION---')
        return 'useful'
      else:
        if self.debug:
          print('---DECISION: GENERATION DOES NOT ADDRESS QUESTION---')
        return 'not useful'
    else:
      if self.debug:
        print('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY---')
      return 'not supported'
