import os
from langchain_core.retrievers import BaseRetriever
from Bio import Entrez, SeqIO
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from langchain.schema import Document

class NCBIProteinRetriever(BaseRetriever):
  k: int

  def __init__(self, k: int):
    super().__init__(k=k)

    self.k = k

    entrez_email = os.getenv('ENTREZ_EMAIL')
    if entrez_email == None:
      raise ValueError('ENTREZ_EMAIL is not defined')
    Entrez.email = entrez_email

  def _search_protein(self, query):
    handle = Entrez.esearch(db='protein', term=query, retmax=self.k)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

  def _fetch_protein(self, protein_id):
    handle = Entrez.efetch(db='protein', id=protein_id, rettype='gb', retmode='text')
    record = SeqIO.read(handle, 'genbank')
    handle.close()
    return record

  def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
    protein_ids = self._search_protein(query)
    docs = []

    for protein_id in protein_ids:
      protein_record = self._fetch_protein(protein_id)
      molecule_type = protein_record.annotations.get("molecule_type", "N/A")
      organism = protein_record.annotations.get("organism", "N/A")
      comment = protein_record.annotations.get("comment", "N/A")
      page_content = (
        f'Protein ID: {protein_id}\n'
        f'Type: {molecule_type}\n'
        f'Name: {protein_record.name}\n'
        f'Organism: {organism}\n'
        f'Description: {protein_record.description}\n'
        f'Comment: {comment}\n'
        f'Sequence: {protein_record.seq}'
      )
      source = f'https://www.ncbi.nlm.nih.gov/protein/{protein_record.name}'
      doc = Document(page_content=page_content, metadata={'source': source})
      docs.append(doc)

    return docs
