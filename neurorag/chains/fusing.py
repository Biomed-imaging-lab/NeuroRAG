from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from neurorag.chains.json_extractor import JsonExtractor


class FusingSchema(BaseModel):
  final_response: str = Field(description='The final fused response.')


template = """Merge multiple expert answers into ONE response that is maximally useful to the user.

RULES:
1. Keep every unique fact, mechanism, name, number, and finding from ALL responses — nothing useful should be lost.
2. If responses disagree, include the most specific/evidence-backed version and note the discrepancy briefly.
3. Remove only exact duplicates and generic filler ("it is worth noting", "in summary", etc.).
4. Match length to query complexity: a simple factual query deserves a concise merged answer; a complex mechanistic query deserves a thorough one. Never pad with generalities.
5. DO NOT add meta-commentary like "Here is the merged answer" or "In conclusion".

Original query:
{query}

Responses to merge:
{responses}

Return STRICTLY valid JSON. Put the full merged answer in the `final_response` field.

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
    ).with_retry(stop_after_attempt=2)

  def invoke(self, data: dict) -> str:
    return self.chain.invoke(data).final_response
