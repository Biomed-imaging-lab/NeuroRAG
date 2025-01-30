from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from json_extractor import JsonExtractor

class HallucinationsSchema(BaseModel):
  binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

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
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )

    self.chain = RunnableParallel(
        completion=prompt | llm | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

  def invoke(self, generation: str, documents: str) -> str:
    return self.chain.invoke({'generation': generation, 'documents': documents})
