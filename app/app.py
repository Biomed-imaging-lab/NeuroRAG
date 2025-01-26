import streamlit as st
import pickle
from neurorag.neurorag import NeuroRAG
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

app = NeuroRAG()
app.compile()

title = '🧠 NeuroRAG Chatbot'

st.set_page_config(page_title=title)

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
    content = response['generation']
    documents = response['documents']
    sources = [document.metadata['source'] for document in documents]
    content += '\n\nSources:\n' + '\n'.join(map(lambda src: f'- {src}', sources))
    st.session_state.messages.append({'role': 'assistant', 'content': content})
    st.chat_message('assistant').write(content)
