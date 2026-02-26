from operator import itemgetter
from typing import TypedDict

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable

from neurorag.chains.fusing import FusingChain, FusingSchema
from neurorag.chains.json_extractor import JsonExtractor
from neurorag.models.OpenRouter import OpenRouter


class FuseData(TypedDict):
  gpt_res: str
  mistral_res: str
  biomistral_res: str
  query: str


template = """
You are a careful scientific/medical assistant. Answer the user's question using ONLY the provided CONTEXT.

### Style requirements (strict)
- Be **fact-focused** and **specific**. No filler, no generic disclaimers, no motivational language.
- Be **thorough and detailed**: include ALL relevant facts, mechanisms, and evidence from CONTEXT. Aim for at least three substantial paragraphs.
- Write in **clear Markdown** with headings and bullet points where helpful.
- Prefer concrete statements, mechanisms, criteria, numbers, and named entities when present in CONTEXT.
- If CONTEXT does not contain enough information to answer some part, say exactly what is missing in a short "### Missing info" section.

### Output format (Markdown)
Use this structure (omit sections that do not apply):
- "### Answer" (direct answer first)
- "### Key facts from context" (bullets; only facts grounded in CONTEXT)
- "### Reasoning / synthesis" (short, connects facts to the answer)
- "### Missing info" (only if needed)

QUERY:
{query}

CONTEXT:
{context}
"""

parser = PydanticOutputParser(pydantic_object=FusingSchema)


class GenerationChain:
  def __init__(self, temperature: float = 0, llms: dict | None = None) -> None:
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
