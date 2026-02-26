import asyncio
import operator
import os
from datetime import datetime
from typing import Annotated, Literal

import nest_asyncio

nest_asyncio.apply()

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import (
  ArxivRetriever,
  BM25Retriever,
  PubMedRetriever,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from neurorag.chains.answer_grade import AnswerGradeChain
from neurorag.chains.biorxiv import BioRxivChain, MedRxivChain
from neurorag.chains.decomposition import DecompositionChain
from neurorag.chains.document_grade import DocumentGradeChain
from neurorag.chains.flare import FlareChain
from neurorag.chains.generation import GenerationChain
from neurorag.chains.hallucinations import HallucinationsChain
from neurorag.chains.hyde import HyDEChain
from neurorag.chains.ncbi_gene import NCBIGeneChain
from neurorag.chains.ncbi_protein import NCBIProteinChain
from neurorag.chains.route import RouteChain
from neurorag.chains.step_back import StepBackChain
from neurorag.models.OpenRouter import OpenRouter
from neurorag.models.OpenRouterEmbeddings import OpenRouterEmbeddings

load_dotenv()

# Chroma cloud configuration
CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
CHROMA_TENANT = os.getenv('CHROMA_TENANT')
CHROMA_DATABASE = os.getenv('CHROMA_DATABASE', 'neurorag')
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'documents')

# Timeouts and concurrency
HYDE_NODE_TIMEOUT = 300  # seconds for all HyDE calls
GRADE_DOCUMENTS_TIMEOUT = 180
RETRIEVER_NODE_TIMEOUT = 120
OPENROUTER_REQUEST_TIMEOUT = 90
OPENROUTER_MAX_RETRIES = 3
MAX_CONCURRENT_LLM_CALLS = 3  # 1 = sequential; avoids OpenRouter 429 and retry deadlock


class GraphStateSchema(TypedDict):
  query: str

  specialized_sources: list[str]

  step_back_query: str
  subqueries: list[str]

  generated_documents: list[str]

  documents: Annotated[list[Document], operator.add]

  web_search: bool
  web_results: list[Document]

  generation: str
  generations_number: int


class NeuroRAG:
  def __init__(
    self,
    model='meta-llama/llama-3.1-8b-instruct',
    embeddings_model='openai/text-embedding-3-small',
    temperature: float = 0,
    debug: bool = False,
    generation_prompt=None,
    max_retries: int = 1,
    llms=None,
    use_flare: bool = False,
  ) -> None:
    self.temperature = temperature
    self.debug = debug
    self.generation_prompt = generation_prompt
    self.max_retries = max_retries
    self.llms = llms
    self.use_flare = use_flare
    self.llm = OpenRouter(
      model=model,
      temperature=self.temperature,
      request_timeout=OPENROUTER_REQUEST_TIMEOUT,
      max_retries=OPENROUTER_MAX_RETRIES,
    )
    self.embeddings_model = embeddings_model

  def _debug_print(self, *args, **kwargs):
    """Print with timestamp if debug mode is enabled."""
    if self.debug:
      timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
      print(f'[{timestamp}]', *args, **kwargs)

  def compile(self) -> None:
    self.embeddings = OpenRouterEmbeddings(model=self.embeddings_model)

    chroma_client = chromadb.CloudClient(
      tenant=CHROMA_TENANT,
      database=CHROMA_DATABASE,
      api_key=CHROMA_API_KEY,
    )
    self.vector_store = Chroma(
      client=chroma_client,
      collection_name=CHROMA_COLLECTION_NAME,
      embedding_function=self.embeddings,
    )
    self.vector_store_retriever = self.vector_store.as_retriever()
    self.pub_med_retriever = PubMedRetriever(top_k_results=3)
    self.arxiv_retriever = ArxivRetriever(load_max_docs=3, get_full_documents=True)

    self.route_chain = RouteChain(self.llm)
    self.hyde_chain = HyDEChain(self.llm)
    self.step_back_chain = StepBackChain(self.llm)
    self.decomposition_chain = DecompositionChain(self.llm)
    self.ncbi_protein_db_chain = NCBIProteinChain(self.llm)
    self.ncbi_gene_db_chain = NCBIGeneChain(self.llm)
    self.biorxiv_chain = BioRxivChain(self.llm)
    self.medrxiv_chain = MedRxivChain(self.llm)
    self.document_grade_chain = DocumentGradeChain(self.llm)
    self.web_search_chain = TavilySearchResults(k=self.max_retries * 3)
    self.generation_chain = GenerationChain(self.temperature, llms=self.llms)
    self.hallucinations_chain = HallucinationsChain(self.llm)
    self.answer_grade_chain = AnswerGradeChain(self.llm)

    if self.use_flare:
      self.flare_chain = FlareChain(
        llm=self.llm,
        retriever_fn=self._flare_retrieve,
      )

    workflow = StateGraph(GraphStateSchema)

    workflow.add_node(
      'determine_specialized_sources', self.determine_specialized_src_node
    )

    workflow.add_node('generate_step_back_query', self.generate_step_back_query_node)
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

    workflow.add_edge(START, 'generate_step_back_query')
    workflow.add_edge('generate_step_back_query', 'generate_subqueries')
    workflow.add_edge('generate_subqueries', 'determine_specialized_sources')

    workflow.add_conditional_edges(
      'determine_specialized_sources',
      self.route_query_node,
      {
        'websearch': 'websearch',
        'specialized_sources': 'generate_hyde_documents',
      },
    )

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

    self._debug_print('---DETERMINE SPECIALIZED SOURCES---')

    try:
      sources = self.route_chain.invoke(query)
      specialized_sources = [source.strip().lower() for source in sources]
    except Exception as e:
      self._debug_print('determine_specialized_src_node', e)
      specialized_sources = []

    self._debug_print(f'---SELECTED SOURCES: {specialized_sources}---')

    return {'specialized_sources': specialized_sources}

  def route_query_node(
    self, state: GraphStateSchema
  ) -> Literal['websearch', 'specialized_sources']:
    sources = state['specialized_sources']

    self._debug_print('---ROUTE QUESTION---')

    return 'websearch' if len(sources) == 0 else 'specialized_sources'

  def generate_step_back_query_node(self, state: GraphStateSchema):
    query = state['query']

    self._debug_print('---GENERATE STEP-BACK QUERY---')

    try:
      step_back_query = self.step_back_chain.invoke(query)
    except Exception as e:
      self._debug_print('generate_step_back_query_node', e)
      step_back_query = query

    return {'step_back_query': step_back_query}

  def generate_subqueries_node(self, state: GraphStateSchema):
    query = state['query']

    self._debug_print('---GENERATE SUBQUERIES---')

    subqueries: list[str] = []

    try:
      subqueries = self.decomposition_chain.invoke(query)
      # Limit to a maximum of 2 subqueries for faster retrieval
      subqueries = subqueries[:2]
    except Exception as e:
      self._debug_print('generate_subqueries_node', e)

    return {'subqueries': subqueries}

  def generate_hyde_documents_node(self, state: GraphStateSchema):
    query = state['query']
    step_back_query = state['step_back_query']
    subqueries = state['subqueries']

    self._debug_print('---GENERATE HYDE DOCUMENTS---')

    queries = [query, step_back_query, *subqueries]

    async def generate_all_hyde_documents():
      sem = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

      async def one(q):
        async with sem:
          return await self.hyde_chain.ainvoke(q)

      return await asyncio.gather(
        *[one(q) for q in queries],
        return_exceptions=False,
      )

    try:
      generated_documents = asyncio.run(
        asyncio.wait_for(
          generate_all_hyde_documents(),
          timeout=HYDE_NODE_TIMEOUT,
        )
      )
    except asyncio.TimeoutError:
      self._debug_print(
        'generate_hyde_documents_node timed out after', HYDE_NODE_TIMEOUT, 's'
      )
      raise RuntimeError(
        f'HyDE generation timed out after {HYDE_NODE_TIMEOUT}s. '
        'Try fewer subqueries or increase HYDE_NODE_TIMEOUT.'
      ) from None

    return {'generated_documents': generated_documents}

  def vector_store_retriever_node(self, state: GraphStateSchema):
    generated_documents = state['generated_documents']
    specialized_sources = state['specialized_sources']

    if 'vectorstore' not in specialized_sources:
      return {'documents': []}

    self._debug_print('---RETRIEVE FROM VECTOR STORE---')

    async def retrieve_all():
      tasks = [self.vector_store_retriever.ainvoke(doc) for doc in generated_documents]
      return await asyncio.gather(*tasks)

    try:
      results = asyncio.run(
        asyncio.wait_for(retrieve_all(), timeout=RETRIEVER_NODE_TIMEOUT)
      )
    except asyncio.TimeoutError:
      self._debug_print('vector_store_retriever_node timed out')
      results = []
    documents = [doc for result in results for doc in result]

    return {'documents': documents}

  def pub_med_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']
    query = state['query']
    step_back_query = state['step_back_query']
    subqueries = state['subqueries']

    if 'pubmed' not in specialized_sources:
      return {'documents': []}

    self._debug_print('---RETRIEVE FROM PUBMED---')

    queries = [query, step_back_query, *subqueries]

    async def retrieve_all():
      async def safe_retrieve(q):
        try:
          return await self.pub_med_retriever.ainvoke(q)
        except Exception as e:
          self._debug_print('pub_med_retriever_node', e)
          return []

      tasks = [safe_retrieve(q) for q in queries]
      return await asyncio.gather(*tasks)

    try:
      results = asyncio.run(
        asyncio.wait_for(retrieve_all(), timeout=RETRIEVER_NODE_TIMEOUT)
      )
    except asyncio.TimeoutError:
      self._debug_print('pub_med_retriever_node timed out')
      results = []
    documents = [doc for result in results for doc in result]

    for document in documents:
      document.metadata['source'] = document.metadata.get('Title', 'PubMed')

    return {'documents': documents}

  def arxiv_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']
    query = state['query']
    step_back_query = state['step_back_query']
    subqueries = state['subqueries']

    if 'arxiv' not in specialized_sources:
      return {'documents': []}

    self._debug_print('---RETRIEVE FROM ARXIV---')

    queries = [query, step_back_query, *subqueries]

    async def retrieve_all():
      async def safe_retrieve(q):
        try:
          return await self.arxiv_retriever.ainvoke(q)
        except Exception as e:
          self._debug_print('arxiv_retriever_node', e)
          return []

      tasks = [safe_retrieve(q) for q in queries]
      return await asyncio.gather(*tasks)

    try:
      results = asyncio.run(
        asyncio.wait_for(retrieve_all(), timeout=RETRIEVER_NODE_TIMEOUT)
      )
    except asyncio.TimeoutError:
      self._debug_print('arxiv_retriever_node timed out')
      results = []
    documents = [doc for result in results for doc in result]

    for document in documents:
      document.metadata['source'] = document.metadata.get('Title', 'arXiv')

    return {'documents': documents}

  def ncbi_protein_db_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'ncbi_protein' not in specialized_sources:
      return {'documents': []}

    self._debug_print('---RETRIEVE FROM NCBI PROTEIN DB---')

    query = state['query']

    try:
      documents = self.ncbi_protein_db_chain.invoke(query)
    except Exception as e:
      self._debug_print('ncbi_protein_db_retriever_node', e)
      documents = []

    return {'documents': documents}

  def ncbi_gene_db_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'ncbi_gene' not in specialized_sources:
      return {'documents': []}

    self._debug_print('---RETRIEVE FROM NCBI GENE DB---')

    query = state['query']

    try:
      documents = self.ncbi_gene_db_chain.invoke(query)
    except Exception as e:
      self._debug_print('ncbi_gene_db_retriever_node', e)
      documents = []

    return {'documents': documents}

  def biorxiv_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'biorxiv' not in specialized_sources:
      return {'documents': []}

    self._debug_print('---RETRIEVE FROM BIORXIV---')

    query = state['query']
    step_back_query = state['step_back_query']
    subqueries = state['subqueries']

    queries = [query, step_back_query, *subqueries]

    async def retrieve_all():
      async def safe_retrieve(q):
        try:
          return await self.biorxiv_chain.ainvoke(q)
        except Exception as e:
          self._debug_print('biorxiv_retriever_node', e)
          return []

      tasks = [safe_retrieve(q) for q in queries]
      return await asyncio.gather(*tasks)

    try:
      results = asyncio.run(
        asyncio.wait_for(retrieve_all(), timeout=RETRIEVER_NODE_TIMEOUT)
      )
    except asyncio.TimeoutError:
      self._debug_print('biorxiv_retriever_node timed out')
      results = []
    documents = [doc for result in results for doc in result]

    return {'documents': documents}

  def medrxiv_retriever_node(self, state: GraphStateSchema):
    specialized_sources = state['specialized_sources']

    if 'medrxiv' not in specialized_sources:
      return {'documents': []}

    self._debug_print('---RETRIEVE FROM MEDRXIV---')

    query = state['query']
    step_back_query = state['step_back_query']
    subqueries = state['subqueries']

    queries = [query, step_back_query, *subqueries]

    async def retrieve_all():
      async def safe_retrieve(q):
        try:
          return await self.medrxiv_chain.ainvoke(q)
        except Exception as e:
          self._debug_print('medrxiv_retriever_node', e)
          return []

      tasks = [safe_retrieve(q) for q in queries]
      return await asyncio.gather(*tasks)

    try:
      results = asyncio.run(
        asyncio.wait_for(retrieve_all(), timeout=RETRIEVER_NODE_TIMEOUT)
      )
    except asyncio.TimeoutError:
      self._debug_print('medrxiv_retriever_node timed out')
      results = []
    documents = [doc for result in results for doc in result]

    return {'documents': documents}

  def grade_documents_node(self, state: GraphStateSchema):
    query = state['query']
    documents = state['documents']

    self._debug_print('---GRADE DOCUMENTs---')

    if len(documents) == 0:
      return {'documents': [], 'web_search': True}

    unique_documents = list({doc.page_content: doc for doc in documents}.values())

    self._debug_print(
      f'---AFTER EXACT DEDUPLICATION: {len(unique_documents)} documents---'
    )

    retriever = BM25Retriever.from_documents(unique_documents, k=10)
    retrieved_documents = retriever.invoke(query)

    self._debug_print(
      f'---BM25 TOP CANDIDATES: {len(retrieved_documents)} documents---'
    )

    # Grade documents with limited concurrency and timeout
    async def grade_all_documents():
      sem = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

      async def grade_single(doc):
        async with sem:
          try:
            grade = await self.document_grade_chain.ainvoke(query, doc)
            return (doc, grade.lower() == 'yes')
          except Exception as e:
            self._debug_print('grade_documents_node', e)
            return (doc, False)

      tasks = [grade_single(doc) for doc in retrieved_documents]
      return await asyncio.gather(*tasks)

    try:
      grading_results = asyncio.run(
        asyncio.wait_for(
          grade_all_documents(),
          timeout=GRADE_DOCUMENTS_TIMEOUT,
        )
      )
    except asyncio.TimeoutError:
      self._debug_print(
        'grade_documents_node timed out after', GRADE_DOCUMENTS_TIMEOUT, 's'
      )
      raise RuntimeError(
        f'Document grading timed out after {GRADE_DOCUMENTS_TIMEOUT}s.'
      ) from None
    filtered_documents = [doc for doc, is_relevant in grading_results if is_relevant]
    filtered_documents = filtered_documents[:3]

    self._debug_print(f'---FINAL DOCUMENTS NUMBER: {len(filtered_documents)}---')

    state['documents'].clear()
    return {
      'documents': filtered_documents,
      'web_search': len(filtered_documents) == 0,
    }

  def decide_to_generate_node(self, state: GraphStateSchema):
    web_search = state['web_search']

    self._debug_print('---ASSESS GRADED DOCUMENTS---')

    if web_search:
      self._debug_print(
        '---DECISION: SOME DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---'
      )
      return 'websearch'
    else:
      self._debug_print('---DECISION: GENERATE---')
      return 'generate'

  def web_search_node(self, state: GraphStateSchema):
    query: str = state['query']
    web_results: list[Document] = state.get('web_results', [])
    generations_number: int = state.get('generations_number', 0)

    self._debug_print('---WEB SEARCH---')

    if not web_results:
      try:
        raw_web_results = self.web_search_chain.invoke(query)
        web_results = [
          Document(page_content=result['content'], metadata={'source': result['url']})
          for result in raw_web_results
        ]
      except Exception as e:
        self._debug_print('web_search_node', e)
        web_results = []

    new_documents = web_results[generations_number * 3 : generations_number * 3 + 3]

    return {'documents': new_documents, 'web_results': web_results}

  def _flare_retrieve(self, query: str) -> list[Document]:
    """Lightweight retriever used by FLARE for on-the-fly retrieval."""
    docs: list[Document] = []
    try:
      docs.extend(self.vector_store_retriever.invoke(query)[:3])
    except Exception:
      pass
    try:
      docs.extend(self.pub_med_retriever.invoke(query)[:2])
    except Exception:
      pass
    return docs

  def generate_node(self, state: GraphStateSchema):
    query = state['query']
    documents = state['documents']
    generations_number = state.get('generations_number', 0)

    self._debug_print('---GENERATE---')

    context = '\n\n'.join(map(lambda doc: doc.page_content, documents))

    if self.use_flare:
      self._debug_print('---USING FLARE GENERATION---')
      generation = self.flare_chain.invoke(query, context)
    else:
      generation = self.generation_chain.invoke(query, context, self.generation_prompt)

    return {'generation': generation, 'generations_number': generations_number + 1}

  def grade_generation_node(
    self, state: GraphStateSchema
  ) -> Literal['useful', 'not useful', 'not supported']:
    query = state['query']
    documents = state['documents']
    generation = state['generation']
    generations_number = state['generations_number']

    self._debug_print('---GRADE GENERATION---')

    if generations_number > self.max_retries:
      return 'useful'

    try:
      context = (
        '\n\n' + '\n\n'.join(map(lambda doc: doc.page_content, documents)) + '\n\n'
      )
      grade = self.hallucinations_chain.invoke(generation, context)
    except Exception as e:
      self._debug_print('grade_generation_node hallucinations', e)
      grade = 'no'

    if grade == 'yes':
      self._debug_print('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---')
      try:
        grade = self.answer_grade_chain.invoke(query, generation)
      except Exception as e:
        self._debug_print('grade_generation_node answer_grade', e)
        grade = 'no'

      if grade == 'yes':
        self._debug_print('---DECISION: GENERATION ADDRESSES QUESTION---')
        return 'useful'
      else:
        self._debug_print('---DECISION: GENERATION DOES NOT ADDRESS QUESTION---')
        return 'not useful'
    else:
      self._debug_print(
        '---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY---'
      )
      return 'not supported'
