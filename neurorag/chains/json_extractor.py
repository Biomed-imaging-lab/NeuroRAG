import re
from langchain_core.runnables import Runnable

class JsonExtractor(Runnable):
  json_pattern = r'\{.*?\}'

  def invoke(self, input_data: str) -> str:
    match = re.search(JsonExtractor.json_pattern, input_data, re.DOTALL)

    if match:
      return match.group().strip().replace('\\\\', '\\')

    return input_data
