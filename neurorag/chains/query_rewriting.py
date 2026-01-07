from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from neurorag.chains.json_extractor import JsonExtractor


class QueryRewritingSchema(BaseModel):
  rewritten_query: str = Field(
    description='Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.'
  )


template = """
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

{format_instructions}

Original query: {query}

Rewritten query:
"""

parser = PydanticOutputParser(pydantic_object=QueryRewritingSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class QueryRewritingChain:
  def __init__(self, llm):
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=3)

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query}).rewritten_query
