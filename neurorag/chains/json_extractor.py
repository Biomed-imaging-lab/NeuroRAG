import re
from langchain_core.runnables import Runnable

class JsonExtractor(Runnable):
  json_pattern = r'\{.*?\}'

  def invoke(self, input_data: str, *args) -> str:
    matches = re.findall(JsonExtractor.json_pattern, input_data, re.DOTALL)

    if matches:
      return matches[-1].strip().replace('\\\\', '\\')

    return input_data
