import os
from io import BytesIO, StringIO
from xml.etree import ElementTree

import httpx
from Bio import Entrez, SeqIO
from langchain_core.callbacks import (
  AsyncCallbackManagerForRetrieverRun,
  CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

NCBI_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

db_params = {
  'gene': {
    'rettype': 'xml',
    'retmode': 'xml',
  },
  'protein': {
    'rettype': 'gb',
    'retmode': 'text',
  },
}


class NCBIRetriever(BaseRetriever):
  db: str
  k: int
  _email: str = ''

  def __init__(self, db: str, k: int) -> None:
    super().__init__(db=db, k=k)

    self.db = db
    self.k = k

    entrez_email = os.getenv('ENTREZ_EMAIL')
    if entrez_email is None:
      raise ValueError('ENTREZ_EMAIL is not defined')
    self._email = entrez_email
    Entrez.email = entrez_email

  def _search(self, term: str) -> list[str]:
    handle = Entrez.esearch(db=self.db, term=term, retmax=self.k)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

  def _fetch(self, ids: list[str]):
    rettype = db_params[self.db]['rettype']
    retmode = db_params[self.db]['retmode']

    handle = Entrez.efetch(db=self.db, id=ids, rettype=rettype, retmode=retmode)
    if self.db == 'gene':
      records = Entrez.read(handle)
    else:
      records = list(SeqIO.parse(handle, rettype))
    handle.close()
    return records

  async def _search_async(self, term: str) -> list[str]:
    params = {
      'db': self.db,
      'term': term,
      'retmax': self.k,
      'retmode': 'xml',
      'email': self._email,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
      response = await client.get(f'{NCBI_BASE_URL}/esearch.fcgi', params=params)
      response.raise_for_status()

    # Parse XML to extract IdList
    root = ElementTree.fromstring(response.text)
    id_list = [id_elem.text for id_elem in root.findall('.//Id') if id_elem.text]
    return id_list

  async def _fetch_async(self, ids: list[str]):
    if not ids:
      return []

    rettype = db_params[self.db]['rettype']
    retmode = db_params[self.db]['retmode']

    params = {
      'db': self.db,
      'id': ','.join(ids),
      'rettype': rettype,
      'retmode': retmode,
      'email': self._email,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
      response = await client.get(f'{NCBI_BASE_URL}/efetch.fcgi', params=params)
      response.raise_for_status()

    if self.db == 'gene':
      # Parse XML gene records using Entrez.read with BytesIO (requires binary mode)
      handle = BytesIO(response.content)
      records = Entrez.read(handle)
      handle.close()
    else:
      # Parse GenBank protein records (text mode is fine)
      handle = StringIO(response.text)
      records = list(SeqIO.parse(handle, rettype))
      handle.close()

    return records

  def _get_gene_document(self, record) -> Document:
    gene_id = record['Entrezgene_track-info']['Gene-track']['Gene-track_geneid']
    gene_symbol = record['Entrezgene_gene']['Gene-ref']['Gene-ref_locus']
    gene_description = record.get('Entrezgene_summary', 'N/A')
    organism_name = record['Entrezgene_source']['BioSource']['BioSource_org'][
      'Org-ref'
    ]['Org-ref_taxname']
    page_content = (
      f'Gene ID: {gene_id}\n'
      f'Gene Symbol: {gene_symbol}\n'
      f'Organism: {organism_name}\n'
      f'Description: {gene_description}'
    )
    source = f'https://www.ncbi.nlm.nih.gov/gene/{gene_id}'
    document = Document(page_content=page_content, metadata={'source': source})
    return document

  def _get_protein_document(self, record) -> Document:
    molecule_type = record.annotations.get('molecule_type', 'N/A')
    organism = record.annotations.get('organism', 'N/A')
    comment = record.annotations.get('comment', 'N/A')
    page_content = (
      f'Protein ID: {record.id}\n'
      f'Type: {molecule_type}\n'
      f'Name: {record.name}\n'
      f'Organism: {organism}\n'
      f'Description: {record.description}\n'
      f'Comment: {comment}\n'
      f'Sequence: {record.seq}'
    )
    source = f'https://www.ncbi.nlm.nih.gov/protein/{record.id}'
    document = Document(page_content=page_content, metadata={'source': source})
    return document

  def _records_to_docs(self, records) -> list[Document]:
    """Convert fetched records to Document objects."""
    docs = []
    for record in records:
      if self.db == 'gene':
        docs.append(self._get_gene_document(record))
      elif self.db == 'protein':
        docs.append(self._get_protein_document(record))
    return docs

  def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
  ) -> list[Document]:
    ids = self._search(query)
    records = self._fetch(ids)
    return self._records_to_docs(records)

  async def _aget_relevant_documents(
    self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
  ) -> list[Document]:
    """Async implementation using httpx for non-blocking I/O."""
    ids = await self._search_async(query)
    records = await self._fetch_async(ids)
    return self._records_to_docs(records)


if __name__ == '__main__':
  import asyncio
  import time

  from dotenv import load_dotenv

  load_dotenv()

  def test_sync():
    # Test gene retriever
    print('\n--- Gene Retriever (BRCA1) ---')
    start = time.perf_counter()
    gene_retriever = NCBIRetriever(db='gene', k=2)
    docs = gene_retriever.invoke('BRCA1')
    elapsed = time.perf_counter() - start
    print(f'Found {len(docs)} documents in {elapsed:.2f}s')
    for doc in docs:
      print(f'\nSource: {doc.metadata["source"]}')
      print(
        doc.page_content[:300] + '...'
        if len(doc.page_content) > 300
        else doc.page_content
      )

    # Test protein retriever
    print('\n--- Protein Retriever (insulin) ---')
    start = time.perf_counter()
    protein_retriever = NCBIRetriever(db='protein', k=2)
    docs = protein_retriever.invoke('insulin human')
    elapsed = time.perf_counter() - start
    print(f'Found {len(docs)} documents in {elapsed:.2f}s')
    for doc in docs:
      print(f'\nSource: {doc.metadata["source"]}')
      print(
        doc.page_content[:300] + '...'
        if len(doc.page_content) > 300
        else doc.page_content
      )

  async def test_async():
    """Test asynchronous retrieval."""
    print('\n' + '=' * 60)
    print('Testing ASYNC retrieval')
    print('=' * 60)

    gene_retriever = NCBIRetriever(db='gene', k=2)
    protein_retriever = NCBIRetriever(db='protein', k=2)

    # Run both retrievers concurrently
    print('\n--- Running gene and protein queries concurrently ---')
    start = time.perf_counter()
    gene_docs, protein_docs = await asyncio.gather(
      gene_retriever.ainvoke('TP53'),
      protein_retriever.ainvoke('hemoglobin human'),
    )
    elapsed = time.perf_counter() - start
    print(f'Concurrent retrieval completed in {elapsed:.2f}s')

    print(f'\nGene results ({len(gene_docs)} docs):')
    for doc in gene_docs:
      print(f'  - {doc.metadata["source"]}')

    print(f'\nProtein results ({len(protein_docs)} docs):')
    for doc in protein_docs:
      print(f'  - {doc.metadata["source"]}')

  # Run tests
  test_sync()
  asyncio.run(test_async())
