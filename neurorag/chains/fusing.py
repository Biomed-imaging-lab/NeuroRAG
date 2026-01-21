from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from neurorag.chains.json_extractor import JsonExtractor


class FusingSchema(BaseModel):
  final_response: str = Field(description='The final fused response.')


template = """
### Task
You are an expert editor and synthesizer. Merge multiple AI-generated responses into ONE best answer.

### Rules (strict)
- Prefer **specific, verifiable facts** and **shared consensus** across responses.
- If responses disagree, **resolve** it by choosing the most defensible claim and briefly note the uncertainty (do not hand-wave).
- Remove filler, repetition, generic advice, and non-answers.
- Keep the final answer **thorough but not verbose**.
- The final answer MUST be **Markdown** (headings + bullet points) and easy to scan.

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

parser = PydanticOutputParser(pydantic_object=FusingSchema)

prompt = PromptTemplate(
  template=template,
  input_variables=['query', 'responses'],
  partial_variables={'format_instructions': parser.get_format_instructions()},
)


class FusingChain:
  def __init__(self):
    llm = ChatOpenAI(model='gpt-4.1', temperature=0)
    self.chain = (
      prompt | llm | StrOutputParser() | JsonExtractor() | parser
    ).with_retry(stop_after_attempt=3)

  def invoke(self, data: dict) -> str:
    return self.chain.invoke(data).final_response
