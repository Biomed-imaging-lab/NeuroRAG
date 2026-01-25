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
- The "vectorstore" retriever contains documents related to neurobiology and medicine. Use the vectorstore for questions on these topics.
- The "pubmed" retriever contains biomedical literature and research articles. It is particularly useful for answering detailed questions about medical research, clinical studies, and scientific discoveries.
- The "arxiv" retriever contains preprints of research papers across various scientific fields, including physics, mathematics, computer science, and biology. Use the arxiv for questions on recent scientific research and theoretical studies in these areas.
- The "biorxiv" retriever contains preprints specifically in the life sciences, including biology, medicine, and related fields. Use the biorxiv for questions on recent biological and medical research preprints.
- The "medrxiv" retriever contains medical preprints specifically focused on clinical medicine, epidemiology, and health sciences. Use the medrxiv for questions on recent medical research, clinical trials, and health-related preprints.
- The "ncbi_protein" retriever contains protein sequence and functional information. Use the NCBI protein DB for questions related to protein sequences, structures, and functions.
- The "ncbi_gene" retriever contains gene sequence and functional information. Use the NCBI gene DB for questions related to gene sequences, structures, and functions.

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
