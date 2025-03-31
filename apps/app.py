import sys
import os
import base64

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'neurorag'))
)
sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'neurorag', 'chains'))
)

import streamlit as st
import warnings
from dotenv import load_dotenv

from neurorag.neurorag import NeuroRAG

warnings.filterwarnings('ignore')
load_dotenv()

app = NeuroRAG()
app.compile()

title = '🧠 NeuroRAG Chatbot'

st.set_page_config(page_title=title, layout='wide')

with st.sidebar:
  st.title(title)

  is_reset_button_disabled = (
    'messages' not in st.session_state or len(st.session_state.messages) <= 1
  )
  if st.button(
    'Reset chat',
    disabled=is_reset_button_disabled,
    use_container_width=True,
  ):
    del st.session_state['messages']

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

chat_column, document_column = st.columns(2, border=True)

with chat_column:
  if 'messages' not in st.session_state:
    st.session_state['messages'] = [
      {'role': 'assistant', 'content': 'How can I help you?', 'response': None}
    ]

  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.markdown(message['content'])

  if prompt := st.chat_input():
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.spinner('Thinking...'):
      response = app.invoke(prompt)
      content: str = response['generation']
      documents = response['documents']
      sources = [
        document.metadata['source']
        for document in documents
        if 'source' in document.metadata
      ]

      if sources:
        content += '\n\nSources:\n' + '\n'.join(map(lambda src: f'- {src}', sources))

      with st.chat_message('assistant'):
        st.markdown(content)
      st.session_state.messages.append(
        {'role': 'assistant', 'content': content, 'response': response}
      )

with document_column:
  assistant_messages = list(
    filter(
      lambda m: m['role'] == 'assistant' and m['response'] is not None,
      st.session_state.messages,
    )
  )
  if len(assistant_messages):
    last_message = assistant_messages[-1]
    response = last_message['response']
    documents = response['documents']
    sources = [
      document.metadata['source']
      for document in documents
      if 'source' in document.metadata
      and not document.metadata['source'].startswith('http')
    ]
    tabs = st.tabs(sources)

    for index, tab in enumerate(tabs):
      with tab:
        source = sources[index]
        file_path = f'../documents/{source}'
        with open(file_path, 'rb') as f:
          base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_iframe = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_iframe, unsafe_allow_html=True)
