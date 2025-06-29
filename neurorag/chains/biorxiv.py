from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.schema import Document

from chains.json_extractor import JsonExtractor
from retrievers.BioRxivRetriever import BioRxivRetriever

CATEGORIES = [
  'neuroscience',
  'bioinformatics',
  'genomics',
  'immunology',
  'microbiology',
  'cancer_biology',
  'cell_biology',
  'genetics',
  'systems_biology',
  'biochemistry',
  'biophysics',
  'ecology',
  'evolutionary_biology',
  'molecular_biology',
  'pathology',
  'pharmacology_and_toxicology',
  'physiology',
  'plant_biology',
  'synthetic_biology',
  'zoology',
]


class BioRxivSchema(BaseModel):
  query: str = Field(description='Optimized search query for BioRxiv.')
  category: str = Field(description='Most appropriate BioRxiv category for this query.')


template = f"""
As an expert in bioinformatics and user query optimization for biological preprint databases, your task is to:
1. Transform user questions into precise and effective queries suitable for the BioRxiv preprint database.
2. Select the most appropriate category from the following list for this query:
{', '.join(CATEGORIES)}

Return your answer as a JSON object with 'query' and 'category' fields.

Original query: {{query}}

{{format_instructions}}
"""

parser = PydanticOutputParser(pydantic_object=BioRxivSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class BioRxivChain:
  def __init__(self, llm) -> None:
    retry_parser = RetryOutputParser.from_llm(
      parser=parser,
      llm=llm,
      max_retries=3,
    )

    self.chain = RunnableParallel(
      completion=prompt | llm | JsonExtractor(), prompt_value=prompt
    ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

  def invoke(self, query: str) -> list[Document]:
    try:
      result = self.chain.invoke({'query': query})
      query, category = result.query, result.category
    except Exception:
      category = None
    retriever = BioRxivRetriever(k=3, category=category)
    return retriever._get_relevant_documents(query, run_manager=None)
