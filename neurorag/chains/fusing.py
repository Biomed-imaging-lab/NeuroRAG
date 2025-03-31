from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from json_extractor import JsonExtractor


class FusingSchema(BaseModel):
  correct_answer: str = Field(
    description='Based on the question and the provided context, choose the most accurate letter among [A, B, C, D].'
  )


template = """
### Instructions

As an expert AI assistant in synthesizing information, your task is to merge multiple AI-generated responses into a single, coherent, and comprehensive answer.

1. **Evaluate Responses:** Analyze each response for reliability, relevance, and commonality.
2. **Identify Common Answers:** Determine the most frequently occurring answer or insight across all responses.
3. **Synthesize Information:** Merge the common answers into a unified response.
4. **Format the Response:** Present the final answer in JSON format, ensuring clarity and coherence.

### Context

Original Query:
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
