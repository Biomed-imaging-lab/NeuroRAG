from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from json_extractor import JsonExtractor

class AnswerGradeSchema(BaseModel):
  binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

template = """
You are a grader assessing whether an answer addresses / resolves a question. \n
Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.

User question:
{query}

LLM generation:
{generation}

{format_instructions}
"""

parser = PydanticOutputParser(pydantic_object=AnswerGradeSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query', 'generation'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)

class AnswerGradeChain:
  def __init__(self, llm):
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )

    self.chain = RunnableParallel(
        completion=prompt | llm | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

  def invoke(self, query: str, generation: str) -> str:
    return self.chain.invoke({'query': query, 'generation': generation}).binary_score
