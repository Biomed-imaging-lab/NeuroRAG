from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from neurorag.chains.json_extractor import JsonExtractor


class FusingSchema(BaseModel):
  final_response: str = Field(description='The final fused response.')


template = """
### Instructions

As an expert AI assistant in synthesizing information, your task is to merge multiple AI-generated responses into a single, coherent, and comprehensive answer.

1. **Evaluate Responses:** Analyze each response for reliability, relevance, and commonality.
2. **Identify Common Answers:** Determine the most frequently occurring answer or insight across all responses.
3. **Synthesize Information:** Merge the common answers into a unified response.
4. **Format the Response:** Present the final answer in JSON format, ensuring clarity and coherence.

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
  def __init__(self):
    llm = ChatOpenAI(model='gpt-4.1', temperature=0)
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=3)

  def invoke(self, data: dict) -> str:
    return self.chain.invoke(data).final_response
