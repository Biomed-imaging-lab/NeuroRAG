import torch
import llm_blender
from operator import itemgetter
from typing import TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from json_extractor import JsonExtractor

class FuseData(TypedDict):
  gpt_res: str
  openbio_res: str
  biomistral_res: str
  query: str

device = (
  'cuda'
  if torch.cuda.is_available()
  else 'mps'
  if torch.backends.mps.is_available()
  else 'cpu'
)
blender = llm_blender.Blender()
blender.loadranker('llm-blender/PairRM', device=device)
blender.loadfuser('llm-blender/gen_fuser_3b', device=device)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {query}

Context:

{context}

Answer:
"""

class GenerationChain:
  def __init__(self, user_template: str | None = None) -> None:
    rag_prompt = PromptTemplate.from_template(user_template or template)

    gpt_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    openbio_llm = Ollama(model='taozhiyuai/openbiollm-llama-3:70b_q2_k', temperature=0)
    biomistral_llm = Ollama(model='cniongolo/biomistral', temperature=0)

    gpt_chain = rag_prompt | gpt_llm | StrOutputParser()
    openbio_chain = rag_prompt | openbio_llm | StrOutputParser()
    biomistral_chain = rag_prompt | biomistral_llm | StrOutputParser()

    self.chain: RunnableSerializable = (
      {
        'gpt_res': gpt_chain,
        'openbio_res': openbio_chain,
        'biomistral_res': biomistral_chain,
        'query': itemgetter('query')
      }
      | RunnableLambda(self.__fuse_generations)
    )

  def __fuse_generations(self, dict: FuseData) -> str:
    query = dict['query']

    gpt_res = dict['gpt_res']
    openbio_res = dict['openbio_res']
    biomistral_res = dict['biomistral_res']
    answers = [gpt_res, openbio_res, biomistral_res]

    fuse_generations, ranks = blender.rank_and_fuse(
      [query],
      [answers],
      instructions=['keep the similar length of the output as the candidates.'],
      return_scores=False,
      batch_size=1,
      top_k=5,
    )
    return fuse_generations[0]

  def invoke(self, query: str) -> str:
    return self.chain.invoke({'query': query})
