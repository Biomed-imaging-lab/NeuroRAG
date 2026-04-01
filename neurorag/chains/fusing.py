from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from neurorag.chains.json_extractor import JsonExtractor
from neurorag.models.OpenRouter import OpenRouter


class FusingSchema(BaseModel):
  final_response: str = Field(description='The final fused response.')


template = """
### Task
You are an expert editor and synthesizer. Merge multiple AI-generated responses into ONE best answer.

### Rules (strict)
- Prefer **specific, verifiable facts** and **shared consensus** across responses.
- If responses disagree, **resolve** it by choosing the most defensible claim and briefly note the uncertainty (do not hand-wave).
- Keep the final answer **thorough and detailed** — preserve ALL unique facts and details from every response. Aim for at least three substantial paragraphs.
{extra_rules}
### Input
Original query:
{query}

Individual responses:
{responses}

### Output
Return STRICTLY valid JSON following these format instructions.
IMPORTANT: Put the Markdown answer inside the `final_response` field exactly.

{format_instructions}
"""

_ARENA_RULE = '- The final answer MUST be **Markdown** (headings + bullet points) and easy to scan.\n'

parser = PydanticOutputParser(pydantic_object=FusingSchema)


class FusingChain:
  def __init__(self, is_for_arena: bool = False, answer_style: str = ''):
    extra_rules = ''
    if is_for_arena:
      extra_rules += _ARENA_RULE
    if answer_style:
      extra_rules += f'- ANSWER STYLE: {answer_style}\n'
    prompt = PromptTemplate(
      template=template,
      input_variables=['query', 'responses'],
      partial_variables={
        'format_instructions': parser.get_format_instructions(),
        'extra_rules': extra_rules,
      },
    )
    llm = OpenRouter(model='openai/gpt-4.1', temperature=0)
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=2)

  def invoke(self, data: dict) -> str:
    return self.chain.invoke(data).final_response
