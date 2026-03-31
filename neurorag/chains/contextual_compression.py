from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


template = """Extract ONLY the parts of the DOCUMENT that are directly relevant to answering the QUERY.

Rules:
- Keep sentences and passages that contain facts, data, mechanisms, or evidence useful for the QUERY.
- Remove boilerplate, author information, irrelevant background, references, and unrelated details.
- Preserve exact wording of kept passages — do not paraphrase or add information.
- If the entire document is relevant, return it unchanged.
- If nothing is relevant, return exactly: EMPTY

QUERY:
{query}

DOCUMENT:
{document}
"""

prompt = PromptTemplate.from_template(template)


class ContextualCompressionChain:
  def __init__(self, llm):
    self.chain = prompt | llm | StrOutputParser()

  def invoke(self, query: str, document: str) -> str:
    result = self.chain.invoke({'query': query, 'document': document}).strip()
    if result == 'EMPTY':
      return ''
    return result

  async def ainvoke(self, query: str, document: str) -> str:
    result = await self.chain.ainvoke({'query': query, 'document': document})
    result = result.strip()
    if result == 'EMPTY':
      return ''
    return result
