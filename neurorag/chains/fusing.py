from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from json_extractor import JsonExtractor


class FusingSchema(BaseModel):
  final_response: str = Field(final_response='The final fused response.')


template = """
### Instructions

As an expert AI assistant in synthesizing information, your task is to merge multiple AI-generated responses into a single, coherent, and comprehensive answer.
Select the most prevalent answer and return it as the final output. Keep the answer verbose, with a minimum of three paragraphs.

### Context

Original query:
{query}

### Individual Responses

{responses}

### Format instructions

- Create a comprehensive, unified response that intelligently merges insights from all sources.
- Ensure the final response is clear, concise, and well-structured in JSON format.
- Highlight the most reliable information while maintaining a cohesive narrative.

{format_instructions}
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
