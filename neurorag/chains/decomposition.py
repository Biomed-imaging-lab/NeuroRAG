from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from neurorag.chains.json_extractor import JsonExtractor


class DecompositionSchema(BaseModel):
  subqueries: list[str] = Field(
    description='Given the original query, decompose it into 2-4 simpler sub-queries as json array of strings'
  )


template = """
You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Original query: {query}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?

{format_instructions}
"""

parser = PydanticOutputParser(pydantic_object=DecompositionSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class DecompositionChain:
  def __init__(self, llm):
    self.chain = (
      prompt
      | llm
      | StrOutputParser()
      | JsonExtractor()
      | parser
    ).with_retry(stop_after_attempt=3)

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query}).subqueries
