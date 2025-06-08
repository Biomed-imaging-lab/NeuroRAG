import os
from operator import itemgetter
from typing import TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from fusing import FusingChain, FusingSchema

from json_extractor import JsonExtractor
from langchain_core.output_parsers import PydanticOutputParser


ollama_server_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


class FuseData(TypedDict):
  gpt_res: str
  mistral_res: str
  biomistral_res: str
  query: str


template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
Keep the answer verbose, with a minimum of three paragraphs.

QUERY: {query}

CONTEXT:
{context}

First, identify the key scientific concepts and data points in the CONTEXT that relate to the QUERY.
Then, analyze how these concepts connect to form a comprehensive answer.
Finally, synthesize your findings into a detailed response.
"""

parser = PydanticOutputParser(pydantic_object=FusingSchema)


class GenerationChain:
  def __init__(self, llm, temperature: float = 0) -> None:
    self.fusing_chain = FusingChain(llm)
    self.gpt_llm = ChatOpenAI(model='gpt-4o', temperature=temperature)
    try:
      self.mistral_llm = ChatMistralAI(
        model='mistral-large-latest',
        temperature=temperature,
      )
    except:
      # Fallback
      self.mistral_llm = Ollama(
        model='mistral-small3.1',
        temperature=temperature,
        bese_url=ollama_server_url,
      )
    self.biomistral_llm = Ollama(
      model='cniongolo/biomistral',
      temperature=temperature,
      bese_url=ollama_server_url,
    )

  def __fuse_responses(self, dict: FuseData, *args):
    query = dict['query']

    gpt_res = dict['gpt_res'].strip()
    mistral_res = dict['mistral_res'].strip()
    biomistral_res = dict['biomistral_res'].strip()

    responses = [gpt_res, mistral_res, biomistral_res]
    responses = list(filter(bool, responses))
    combined_responses = '\n\n--------\n\n'.join(responses)

    try:
      fused_response = self.fusing_chain.invoke(
        {'query': query, 'responses': combined_responses}
      )
      return fused_response
    except Exception:
      try:
        return (JsonExtractor() | parser).invoke(responses[0]).correct_answer
      except Exception:
        return responses[0]

  def invoke(self, query: str, context: str, user_prompt=None) -> str:
    rag_prompt = user_prompt or PromptTemplate.from_template(template)

    gpt_chain = rag_prompt | self.gpt_llm | StrOutputParser()
    mistral_chain = rag_prompt | self.mistral_llm | StrOutputParser()
    biomistral_chain = rag_prompt | self.biomistral_llm | StrOutputParser()

    chain: RunnableSerializable = {
      'query': itemgetter('query'),
      'gpt_res': gpt_chain,
      'mistral_res': mistral_chain,
      'biomistral_res': biomistral_chain,
    } | RunnableLambda(self.__fuse_responses)

    return chain.invoke({'query': query, 'context': context})
