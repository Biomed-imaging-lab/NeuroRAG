from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from json_extractor import JsonExtractor

class DocumentGradeSchema(BaseModel):
  binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

template = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

{format_instructions}

User question:
{query}

Retrieved document:
{document}
"""

parser = PydanticOutputParser(pydantic_object=DocumentGradeSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['document', 'query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)

class DocumentGradeChain:
  def __init__(self, llm):
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )

    self.chain = RunnableParallel(
        completion=prompt | llm | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

  def invoke(self, query: str, document: str) -> str:
    return self.chain.invoke({'query': query, 'document': document}).binary_score
