import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neurorag import NeuroRAG

def load_docs():
  with open('splitted_docs.pkl', 'rb') as f:
    return pickle.load(f)

docs = load_docs()
neuro_rag = NeuroRAG(docs)
neuro_rag.compile()

app = FastAPI()

class Question(BaseModel):
  query: str

@app.post('/invoke')
async def invoke(question: Question):
  if not question.query.strip():
    raise HTTPException(status_code=400, detail='Question text cannot be empty.')

  try:
    answer = app.invoke(question.query)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

  return {'answer': answer}
