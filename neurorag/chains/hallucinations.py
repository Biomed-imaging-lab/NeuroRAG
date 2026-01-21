from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from neurorag.chains.json_extractor import JsonExtractor


class HallucinationsSchema(BaseModel):
  binary_score: str = Field(
    description="Answer is grounded in the facts, 'yes' or 'no'"
  )


template = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."

{format_instructions}

Set of facts:
{documents}

LLM generation:
{generation}
"""

parser = PydanticOutputParser(pydantic_object=HallucinationsSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query', 'documents'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class HallucinationsChain:
  def __init__(self, llm):
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=2)

  def invoke(self, generation: str, documents: str) -> str:
    return self.chain.invoke(
      {'generation': generation, 'documents': documents}
    ).binary_score
