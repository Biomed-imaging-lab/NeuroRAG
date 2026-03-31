from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from neurorag.chains.json_extractor import JsonExtractor


class RerankSchema(BaseModel):
  score: int = Field(description='Relevance score from 0 to 10')


template = """Rate how relevant the DOCUMENT is to the QUERY on a scale of 0-10.

Scoring guide:
- 0: completely irrelevant
- 1-3: tangentially related, shares some keywords but doesn't help answer the query
- 4-6: partially relevant, contains some useful information
- 7-9: highly relevant, directly addresses the query with specific facts
- 10: perfectly relevant, directly and comprehensively answers the query

QUERY:
{query}

DOCUMENT:
{document}

{format_instructions}
"""

parser = PydanticOutputParser(pydantic_object=RerankSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query', 'document'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class RerankChain:
  def __init__(self, llm):
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=2)

  def invoke(self, query: str, document: str) -> int:
    return self.chain.invoke({'query': query, 'document': document}).score

  async def ainvoke(self, query: str, document: str) -> int:
    result = await self.chain.ainvoke({'query': query, 'document': document})
    return result.score
