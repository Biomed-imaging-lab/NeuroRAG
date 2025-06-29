import requests
from datetime import datetime, timedelta

from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever


class BioRxivRetriever(BaseRetriever):
  """BioRxiv retriever for searching preprints (local keyword filter)."""

  base_url: str = 'https://api.biorxiv.org/details/biorxiv'
  k: int = 5
  max_results: int = 100  # API returns up to 100 per call
  days_back: int = 30  # How many days back to fetch
  category: str | None = None

  def __init__(
    self,
    k: int = 5,
    max_results: int = 100,
    days_back: int = 30,
    category: str | None = None,
  ) -> None:
    super().__init__()
    self.k = k
    self.max_results = max_results
    self.days_back = days_back
    self.category = category

  def _fetch_recent_papers(self) -> list[dict]:
    """Fetch recent papers from BioRxiv using the interval endpoint, supporting pagination."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=self.days_back)
    interval = f'{start_date.strftime("%Y-%m-%d")}/{end_date.strftime("%Y-%m-%d")}'
    all_papers: list[dict] = []
    cursor = 0
    page_size = 100
    try:
      while len(all_papers) < self.max_results:
        url = f'{self.base_url}/{interval}/{cursor}'
        if self.category:
          url += f'?category={self.category}'
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'collection' in data:
          papers = data['collection']
          if not papers:
            break
          all_papers.extend(papers)
          if len(papers) < page_size:
            break  # Last page
          cursor += page_size
        else:
          break
      return all_papers[: self.max_results]
    except Exception as e:
      print(f'Error fetching BioRxiv papers: {e}')
      return []

  def _filter_by_query(self, papers: list[dict], query: str) -> list[dict]:
    """Filter papers by query in title or abstract (case-insensitive)."""
    query_lower = query.lower()
    filtered = [
      paper
      for paper in papers
      if query_lower in paper.get('title', '').lower()
      or query_lower in paper.get('abstract', '').lower()
    ]
    return filtered

  def _format_document(self, preprint: dict) -> Document:
    title = preprint.get('title', 'No title available')
    authors = preprint.get('authors', 'Unknown authors')
    abstract = preprint.get('abstract', 'No abstract available')
    doi = preprint.get('doi', 'No DOI available')
    date = preprint.get('date', 'Unknown date')
    category = preprint.get('category', 'Unknown category')
    page_content = f"""Title: {title}\nAuthors: {authors}\nCategory: {category}\nDate: {date}\nAbstract: {abstract}"""
    metadata = {
      'source': f'https://doi.org/{doi}',
      'title': title,
      'authors': authors,
      'doi': doi,
      'date': date,
      'category': category,
      'type': 'biorxiv_preprint',
    }
    return Document(page_content=page_content, metadata=metadata)

  def _get_relevant_documents(
    self,
    query: str,
    *,
    run_manager: CallbackManagerForRetrieverRun,
  ) -> list[Document]:
    papers = self._fetch_recent_papers()
    documents = [self._format_document(p) for p in papers]
    if not documents:
      return []
    # Use BM25Retriever for local ranking
    bm25 = BM25Retriever.from_documents(documents)
    top_docs = bm25.invoke(query)
    return top_docs[: self.k]
