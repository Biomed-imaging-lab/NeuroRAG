from operator import itemgetter
from typing import TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from fusing import FusingChain

from json_extractor import JsonExtractor

class FuseData(TypedDict):
  gpt_res: str
  openbio_res: str
  mistral_res: str
  query: str

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {query}

Context:

{context}

Answer:
"""

class GenerationChain:
  def __init__(self, llm, temperature: float = 0, user_prompt = None) -> None:
    rag_prompt = user_prompt or PromptTemplate.from_template(template)

    gpt_llm = ChatOpenAI(model='gpt-4o', temperature=temperature)
    openbio_llm = Ollama(model='taozhiyuai/openbiollm-llama-3:70b_q2_k', temperature=temperature)
    mistral_llm = ChatMistralAI(model='mistral-large-latest', temperature=temperature)

    gpt_chain = rag_prompt | gpt_llm | StrOutputParser()
    openbio_chain = rag_prompt | openbio_llm | StrOutputParser()
    mistral_chain = rag_prompt | mistral_llm | StrOutputParser()

    fusing_chain = FusingChain(llm)

    self.chain: RunnableSerializable = (
      {
        'query': itemgetter('query'),
        'gpt_res': gpt_chain,
        'openbio_res': openbio_chain,
        'mistral_res': mistral_chain,
      } | RunnableLambda(self.__combine_responses)
      | fusing_chain
      | RunnableLambda(self.__extract_final_response)
    )

  def __combine_responses(self, dict: FuseData):
    query = dict['query']

    gpt_res = dict['gpt_res']
    openbio_res = dict['openbio_res']
    mistral_res = dict['mistral_res']

    responses = [gpt_res, openbio_res, mistral_res]
    combined_responses = '\n'.join([f'Response:\n{r}' for r in responses])

    return {'query': query, 'responses': combined_responses}

  def __extract_final_response(self, response) -> str:
    return response.final_response

  def invoke(self, query: str, context: str, user_prompt = None) -> str:
    if user_prompt:
      self.user_prompt = user_prompt

    return self.chain.invoke({'query': query, 'context': context})
