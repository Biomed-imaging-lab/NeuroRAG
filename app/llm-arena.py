import streamlit as st
import warnings
import pandas as pd
import json
import random
from dotenv import load_dotenv
from st_draggable_list import DraggableList
from langchain_community.llms import Ollama
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

st.title('🏟️ LLM-арена')
st.write("""
Здравствуйте! Вы увидите ответы, сгенерированные различными нейронными сетями на
заданные вопросы. Ваша задача — расположить эти ответы в порядке от самого
лучшего (вверху списка) до самого худшего (внизу списка). Для каждой пары
вопрос-ответ вы сможете выбрать уникальный ранг для каждого ответа, чтобы
отразить ваше предпочтение. После того как вы расставите ранги для всех
ответов, нажмите на кнопку "Сохранить", чтобы сохранить ваши оценки в
файл. Это позволит нам проанализировать, какие ответы вы считаете
наиболее качественными.
""")

grades = {}

for index, row in st.session_state['answers_df'].iterrows():
  question = row['question']
  llama = row['llama']
  gpt = row['gpt']

  st.subheader(f'Вопрос {index + 1}: {question}')

  answers = [
    {'id': 'llama', 'order': 1, 'name': llama},
    {'id': 'gpt', 'order': 3, 'name': gpt},
  ]
  random.shuffle(answers)

  slist = DraggableList(answers, width='100%', key=question)
  grades[question] = slist

cols = st.columns(4)
if cols[-1].button('Save', type='primary', use_container_width=True):
  with open('grades.json', 'w') as f:
    json.dump(grades, f)
  st.success('Rankings saved successfully!')
