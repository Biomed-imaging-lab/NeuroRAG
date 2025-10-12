from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from neurorag.chains.json_extractor import JsonExtractor


class RouteSchema(BaseModel):
  sources: list[str] = Field(
    description='Given a user question select the retrieval methods you consider the most appropriate for addressing this question. You may also return an empty array if no methods are required.',
  )


template = """
You are an expert at selecting retrieval methods.
Given a user question select the retrieval methods you consider the most appropriate for addressing user question.
You may also return an empty array if no methods are required.

Possible retrieval methods:
- The "vectorstore" retriever contains documents related to neurobiology and medicine. Use the vectorstore for questions on these topics.
- The "pubmed" retriever contains biomedical literature and research articles. It is particularly useful for answering detailed questions about medical research, clinical studies, and scientific discoveries.
- The "arxiv" retriever contains preprints of research papers across various scientific fields, including physics, mathematics, computer science, and biology. Use the arxiv for questions on recent scientific research and theoretical studies in these areas.
- The "biorxiv" retriever contains preprints specifically in the life sciences, including biology, medicine, and related fields. Use the biorxiv for questions on recent biological and medical research preprints.
- The "medrxiv" retriever contains medical preprints specifically focused on clinical medicine, epidemiology, and health sciences. Use the medrxiv for questions on recent medical research, clinical trials, and health-related preprints.
- The "ncbi_protein" retriever contains protein sequence and functional information. Use the NCBI protein DB for questions related to protein sequences, structures, and functions.
- The "ncbi_gene" retriever contains gene sequence and functional information. Use the NCBI gene DB for questions related to gene sequences, structures, and functions.

{format_instructions}

User question:
{query}
"""

parser = PydanticOutputParser(pydantic_object=RouteSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class RouteChain:
  def __init__(self, llm):
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )

    self.chain = RunnableParallel(
      completion=prompt | llm | StrOutputParser() | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query}).sources
