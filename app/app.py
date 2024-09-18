import streamlit as st
import pickle
from biorag import BioRAG
import warnings

warnings.filterwarnings('ignore')

title = '🧠 BioRAG Chatbot'

st.set_page_config(page_title=title)

def load_docs():
  with open('splitted_docs.pkl', 'rb') as f:
    return pickle.load(f)

docs = load_docs()
app = BioRAG(docs)
app.compile()

with st.sidebar:
  st.title(title)
  st.subheader('Parameters')

  temperature = st.sidebar.slider(
    'temperature',
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
    disabled=True,
  )

  docs_col = st.sidebar.selectbox(
    'Documents collection',
    options=['Neurobiology'],
    disabled=True,
  )

if 'messages' not in st.session_state:
  st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you?'}]

for msg in st.session_state.messages:
  st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input():
  st.session_state.messages.append({'role': 'user', 'content': prompt})
  st.chat_message('user').write(prompt)

  with st.spinner('Thinking...'):
    response = app.invoke(prompt)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    st.chat_message('assistant').write(response)
