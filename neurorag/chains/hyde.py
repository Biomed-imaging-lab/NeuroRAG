from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

template = """
Write a concise scientific passage (2-3 sentences maximum) that answers the question. Focus on key facts and avoid unnecessary details.

Question: {query}

Passage:
"""

prompt = ChatPromptTemplate.from_template(template)


class HyDEChain:
  def __init__(self, llm):
    self.chain = prompt | llm | StrOutputParser()

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query})

  async def ainvoke(self, query: str) -> str:
    return await self.chain.ainvoke({'query': query})
