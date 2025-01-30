from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.schema import Document

from json_extractor import JsonExtractor
from retrievers.NCBIRetriever import NCBIRetriever

class NCBIProteinSchema(BaseModel):
  query: str = Field(description='Given the original query, please find a protein locus for the NCBI protein database.')

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
    retriever = NCBIRetriever(db='protein', k=3)

    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )

    self.chain = RunnableParallel(
        completion=prompt | llm | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x)) | self.__query_extractor | retriever

  def __query_extractor(self, response: NCBIProteinSchema) -> str:
    return response.query

  def invoke(self, query: str) -> list[Document]:
    return self.chain.invoke({'query': query})
