from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from neurorag.chains.json_extractor import JsonExtractor


class StepBackSchema(BaseModel):
  step_back: str = Field(
    description='Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.'
  )


template = """
You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

{format_instructions}

Original query: {query}

Step-back query:
"""

parser = PydanticOutputParser(pydantic_object=StepBackSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class StepBackChain:
  def __init__(self, llm):
    self.chain = (
      prompt
      | llm
      | StrOutputParser()
      | JsonExtractor()
      | parser
    ).with_retry(stop_after_attempt=3)

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query}).step_back
