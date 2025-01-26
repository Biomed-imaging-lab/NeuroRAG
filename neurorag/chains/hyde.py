from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

template = """
Please write a scientific paper passage to answer the question

Question: {query}

Passage:
"""

prompt = ChatPromptTemplate.from_template(template)

class HyDEChain:
  def __init__(self, llm):
    self.chain = prompt | llm | StrOutputParser()

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query})
