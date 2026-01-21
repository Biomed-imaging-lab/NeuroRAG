from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from neurorag.chains.json_extractor import JsonExtractor
from neurorag.retrievers.NCBIRetriever import NCBIRetriever


class NCBIProteinSchema(BaseModel):
  query: str = Field(
    description='Given the original query, please find a protein locus for the NCBI protein database.'
  )


template = """
As an expert in bioinformatics and user query optimization for biological databases, your task is to transform user questions into precise and effective queries suitable for the NCBI protein database.
Create a query with only locus of a protein for search within the NCBI protein database.

Original query: {query}

{format_instructions}
"""

parser = PydanticOutputParser(pydantic_object=NCBIProteinSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class NCBIProteinChain:
  def __init__(self, llm) -> None:
    self.retriever = NCBIRetriever(db='protein', k=3)

    self.parse_chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=2)

  def invoke(self, query: str) -> list[Document]:
    result = self.parse_chain.invoke({'query': query})
    return self.retriever.invoke(result.query)
