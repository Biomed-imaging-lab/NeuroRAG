import os
from operator import itemgetter
from typing import TypedDict, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from neurorag.chains.fusing import FusingChain, FusingSchema
from neurorag.chains.json_extractor import JsonExtractor
from langchain_core.output_parsers import PydanticOutputParser
from neurorag.models.OpenRouter import OpenRouter


ollama_server_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')


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
  def __init__(self, llm, temperature: float = 0, llms: Optional[dict] = None) -> None:
    self.fusing_chain = FusingChain()
    self.temperature = temperature
    self.llms = llms or {
      'openai': OpenRouter(model='openai/gpt-4.1', temperature=temperature),
      'mistral': OpenRouter(model='mistralai/mistral-large', temperature=temperature),
      'claude': OpenRouter(
        model='anthropic/claude-3.5-sonnet', temperature=temperature
      ),
    }

  def __fuse_responses(self, responses_dict, *args):
    query = responses_dict['query']
    responses = []
    for name in responses_dict['llm_responses']:
      res = responses_dict['llm_responses'][name].strip()
      if res:
        responses.append(res)
    combined_responses = '\n\n--------\n\n'.join(responses)
    try:
      fused_response = self.fusing_chain.invoke(
        {'query': query, 'responses': combined_responses}
      )
      return fused_response
    except Exception:
      try:
        return (
          (StrOutputParser() | JsonExtractor() | parser)
          .invoke(responses[0])
          .correct_answer
        )
      except Exception:
        return responses[0]

  def invoke(self, query: str, context: str, user_prompt=None) -> str:
    rag_prompt = user_prompt or PromptTemplate.from_template(template)
    chains = {}
    for name in self.llms:
      chains[name] = rag_prompt | self.llms[name] | StrOutputParser()
    chain: RunnableSerializable = {
      'query': itemgetter('query'),
      'llm_responses': {name: chains[name] for name in chains},
    } | RunnableLambda(self.__fuse_responses)
    return chain.invoke({'query': query, 'context': context})
