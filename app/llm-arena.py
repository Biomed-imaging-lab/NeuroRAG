import streamlit as st
import warnings
import pandas as pd
import json
import markdown
from dotenv import load_dotenv
from st_draggable_list import DraggableList
from langchain_community.llms import Ollama
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings('ignore')
load_dotenv()

llama_chain = Ollama(model='llama3.1', temperature=0) | StrOutputParser()
gpt_chain = ChatOpenAI(model='gpt-4o', temperature=0) | StrOutputParser()

questions_df = pd.read_csv('questions.csv')

if 'answers_df' not in st.session_state:
  answers_df = pd.DataFrame()

  for question in questions_df['question']:
    llama_res = llama_chain.invoke(question)
    gpt_res = gpt_chain.invoke(question)

    row = pd.DataFrame({
      'question': question,
      'llama': llama_res,
      'gpt': gpt_res,
    }, index=[0])
    answers_df = pd.concat([answers_df, row], ignore_index=True)

  st.session_state['answers_df'] = answers_df

title = 'LLM-арена'

st.set_page_config(page_title=title)

grades = {}

for index, row in st.session_state['answers_df'].iterrows():
  question = row['question']
  llama = row['llama']
  gpt = row['gpt']

  st.subheader(f'Question {index + 1}: {question}')

  answers = [
    {'id': 'llama', 'order': 1, 'name': markdown.markdown(llama)},
    {'id': 'gpt', 'order': 3, 'name': markdown.markdown(gpt)},
  ]

  slist = DraggableList(answers, width='100%')
  grades[question] = slist

cols = st.columns(4)
if cols[-1].button('Save', type='primary', use_container_width=True):
  with open('grades.json', 'w') as f:
    json.dump(grades, f)
  st.success('Rankings saved successfully!')
