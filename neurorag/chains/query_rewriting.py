from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from json_extractor import JsonExtractor

class QueryRewritingSchema(BaseModel):
  rewritten_query: str = Field(description='Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.')

template = """
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

{format_instructions}

Original query: {query}

Rewritten query:
"""

parser = PydanticOutputParser(pydantic_object=QueryRewritingSchema)

class QueryRewritingChain:
  def __init__(self, llm):
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )
    prompt = PromptTemplate(
      template=template,
      input_variables=['query'],
      partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    self.chain = RunnableParallel(
        completion=prompt | llm | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query})
