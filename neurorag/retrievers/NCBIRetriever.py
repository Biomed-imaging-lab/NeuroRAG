import os
from Bio import Entrez, SeqIO

from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

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

  def __init__(self, db: str, k: int) -> None:
    super().__init__(db=db, k=k)

    self.db = db
    self.k = k

    entrez_email = os.getenv('ENTREZ_EMAIL')
    if entrez_email == None:
      raise ValueError('ENTREZ_EMAIL is not defined')
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
      records = [SeqIO.read(handle, rettype)]
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

  def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
  ) -> list[Document]:
    ids = self._search(query)
    records = self._fetch(ids)

    docs = []

    for record in records:
      if self.db == 'gene':
        docs.append(self._get_gene_document(record))
      elif self.db == 'protein':
        docs.append(self._get_protein_document(record))

    return docs
