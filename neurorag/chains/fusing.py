from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from json_extractor import JsonExtractor

class FusingSchema(BaseModel):
  final_response: str = Field(description="The final fused response")

template = """
You are an AI assistant tasked with combining multiple AI responses into a single, coherent answer.
Merge the responses intelligently, keeping the most reliable information.
Create a comprehensive, unified response that combines the best insights from all sources in json format.

{format_instructions}

Original query: {query}

Individual responses:
{responses}
"""

parser = PydanticOutputParser(pydantic_object=FusingSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query', 'responses'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)

class FusingChain:
  def __init__(self, llm):
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )

    self.chain = RunnableParallel(
        completion=prompt | llm | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

  def invoke(self, data: dict) -> str:
    return self.chain.invoke(data).final_response
