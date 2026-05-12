from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from neurorag.chains.json_extractor import JsonExtractor
from neurorag.models.OpenRouter import OpenRouter


class FusingSchema(BaseModel):
  final_response: str = Field(description='The final fused response.')


_TEMPLATE_DETAILED = """
### Task
You are an expert editor and synthesizer. Merge multiple AI-generated responses into ONE best answer.

### Rules (strict)
- Prefer **specific, verifiable facts** and **shared consensus** across responses.
- If responses disagree, **resolve** it by choosing the most defensible claim and briefly note the uncertainty.
- Keep the final answer thorough — preserve all unique, non-redundant facts.
{extra_rules}
### Input
Original query:
{query}

Individual responses:
{responses}

### Output
Return STRICTLY valid JSON following these format instructions.
IMPORTANT: Put the answer inside the `final_response` field exactly.

{format_instructions}
"""

_TEMPLATE_CONCISE = """
### Task
You are an expert editor. Pick the single BEST response from the candidates below and compress it to match the OUTPUT RULE exactly.

### OUTPUT RULE (hard limit — overrides everything else)
{answer_style}

### Rules
- Choose the most accurate and factual candidate.
- Remove all redundancy, caveats, and background context.
- Keep only the core fact(s) needed to answer the query.

### Input
Original query:
{query}

Candidate responses:
{responses}

### Output
Return STRICTLY valid JSON. Put the compressed answer inside `final_response`.

{format_instructions}
"""

_ARENA_RULE = '- The final answer MUST be **Markdown** (headings + bullet points) and easy to scan.\n'

parser = PydanticOutputParser(pydantic_object=FusingSchema)


class FusingChain:
  def __init__(self, is_for_arena: bool = False, answer_style: str = ''):
    if answer_style:
      template = _TEMPLATE_CONCISE
      partial_vars = {
        'format_instructions': parser.get_format_instructions(),
        'answer_style': answer_style,
      }
      input_vars = ['query', 'responses']
    else:
      extra_rules = _ARENA_RULE if is_for_arena else ''
      template = _TEMPLATE_DETAILED
      partial_vars = {
        'format_instructions': parser.get_format_instructions(),
        'extra_rules': extra_rules,
      }
      input_vars = ['query', 'responses']
    prompt = PromptTemplate(
      template=template,
      input_variables=input_vars,
      partial_variables=partial_vars,
    )
    llm = OpenRouter(model='openai/gpt-4.1', temperature=0)
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=2)

  def invoke(self, data: dict) -> str:
    return self.chain.invoke(data).final_response
