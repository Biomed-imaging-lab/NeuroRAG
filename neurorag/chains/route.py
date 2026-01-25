from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from neurorag.chains.json_extractor import JsonExtractor


Source = Literal[
  'vectorstore',
  'pubmed',
  'arxiv',
  'biorxiv',
  'medrxiv',
  'ncbi_protein',
  'ncbi_gene',
]


class RouteSchema(BaseModel):
  sources: list[Source] = Field(
    description='Given a user question select the retrieval methods you consider the most appropriate for addressing this question. You may also return an empty array if no methods are required.',
  )


template = """
You are an expert at selecting retrieval methods.
Given a user question select the retrieval methods you consider the most appropriate for addressing user question.
You may also return an empty array if no methods are required.

Possible retrieval methods:
- "vectorstore": Documents about neurobiology and medicine. Use for general medical/neuro questions.
- "pubmed": Biomedical literature. Use for medical research, clinical studies, scientific discoveries.
- "arxiv": Scientific preprints. Use for recent research in physics, math, CS, biology.
- "biorxiv": Life sciences preprints. Use for recent biological research.
- "medrxiv": Medical preprints. Use for clinical medicine, epidemiology, health sciences.
- "ncbi_protein": Protein sequences database. Use ONLY when query explicitly mentions a specific protein name, protein ID, or asks about protein sequence/structure of a named protein (e.g., "hemoglobin structure", "P53 protein sequence").
- "ncbi_gene": Gene sequences database. Use ONLY when query explicitly mentions a specific gene name, gene symbol, locus, or accession number (e.g., "BRCA1 gene", "TP53 mutations", "gene Chr17:1234").

{format_instructions}

User question:
{query}
"""

parser = PydanticOutputParser(pydantic_object=RouteSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class RouteChain:
  def __init__(self, llm):
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=2)

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query}).sources
