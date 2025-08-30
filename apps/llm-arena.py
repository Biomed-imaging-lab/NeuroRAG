import sys

sys.path.append('..')
sys.path.append('../neurorag')
sys.path.append('../neurorag/chains')

import streamlit as st
import warnings
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Optional

from neurorag.neurorag import NeuroRAG
from neurorag.models.OpenRouter import OpenRouter

warnings.filterwarnings('ignore')
load_dotenv()

if 'comparison_history' not in st.session_state:
  st.session_state.comparison_history = []

# Available OpenRouter models
OPENROUTER_MODELS = {
  'Claude 3.5 Sonnet': 'anthropic/claude-3.5-sonnet',
  'GPT-4o': 'openai/gpt-4o',
  'GPT-4o Mini': 'openai/gpt-4o-mini',
  'Claude 3.5 Haiku': 'anthropic/claude-3.5-haiku',
  'Mistral Large': 'mistralai/mistral-large-latest',
  'Llama 3.1 8B': 'meta-llama/llama-3.1-8b-instruct',
  'Llama 3.1 70B': 'meta-llama/llama-3.1-70b-instruct',
  'Gemini Pro': 'google/gemini-pro',
  'CodeLlama 70B': 'meta-llama/codellama-70b-instruct',
  'Qwen2.5 72B': 'qwen/qwen2.5-72b-instruct',
}


def get_openrouter_llm(model_name: str) -> Optional[Any]:
  """Get OpenRouter LLM instance"""
  try:
    llm = OpenRouter(model=model_name, temperature=0, max_tokens=2000)
    return llm | StrOutputParser()
  except Exception as e:
    st.error(f'Error initializing OpenRouter model: {e}')
    return None


def get_neurorag_answer(question: str) -> str:
  """Get answer from NeuroRAG"""
  try:
    neurorag = NeuroRAG(model='llama3.1', temperature=0, debug=False)
    neurorag.compile()
    result = neurorag.invoke(question)
    return 'test placeholder'
    return result.get('generation', 'No answer generated')
  except Exception as e:
    st.error(f'Error getting NeuroRAG answer: {e}')
    return 'Error occurred while generating NeuroRAG answer'


def get_competitor_answer(question: str, model_name: str) -> str:
  """Get answer from competitor model"""
  try:
    llm = get_openrouter_llm(model_name)
    if llm:
      return llm.invoke(question)
    return 'Error: Could not initialize model'
  except Exception as e:
    st.error(f'Error getting competitor answer: {e}')
    return 'Error occurred while generating competitor answer'


def save_comparison(
  question: str,
  neurorag_answer: str,
  competitor_answer: str,
  competitor_model: str,
  user_choice: str,
) -> None:
  """Save comparison result to session state"""
  comparison = {
    'question': question,
    'neurorag_answer': neurorag_answer,
    'competitor_answer': competitor_answer,
    'competitor_model': competitor_model,
    'user_choice': user_choice,
    'timestamp': str(pd.Timestamp.now()),
  }
  st.session_state.comparison_history.append(comparison)


def export_results() -> bool:
  """Export comparison results to JSON file"""
  if st.session_state.comparison_history:
    with open('llm_arena_results.json', 'w') as f:
      json.dump(st.session_state.comparison_history, f, indent=2)
    return True
  return False


st.set_page_config(page_title='NeuroRAG LLM Arena', page_icon='🏟️', layout='wide')

st.title('🏟️ NeuroRAG LLM Arena')
st.markdown("""
Compare NeuroRAG's performance against other AI models. Ask a question and see how both models respond, then vote on which answer is better.
""")

with st.sidebar:
  st.header('⚙️ Settings')

  selected_model = st.selectbox(
    'Choose competitor model:', list(OPENROUTER_MODELS.keys()), index=0
  )

  if st.button('📊 Export Results', type='secondary'):
    if export_results():
      st.success('Results exported to llm_arena_results.json')
    else:
      st.warning('No results to export')

  # Show comparison history
  if st.session_state.comparison_history:
    st.header('📝 History')
    for i, comp in enumerate(st.session_state.comparison_history[-5:]):  # Show last 5
      with st.expander(f'Q{i + 1}: {comp["question"][:50]}...'):
        st.write(f'**Model:** {comp["competitor_model"]}')
        st.write(f'**Choice:** {comp["user_choice"]}')

question = st.text_area(
  'Enter your question:',
  placeholder="Ask anything you'd like to compare between NeuroRAG and the selected model...",
  height=100,
)

if st.button('Generate Answers', type='primary', disabled=not question.strip()):
  if question.strip():
    with st.spinner('Generating answers...'):
      neurorag_answer = get_neurorag_answer(question)
      competitor_answer = get_competitor_answer(
        question, OPENROUTER_MODELS[selected_model]
      )

      st.session_state.current_question = question
      st.session_state.neurorag_answer = neurorag_answer
      st.session_state.competitor_answer = competitor_answer
      st.session_state.competitor_model = selected_model

if hasattr(st.session_state, 'current_question') and st.session_state.current_question:
  col1, col2 = st.columns(2)

  with col1:
    st.subheader('🧠 NeuroRAG')
    st.markdown('**Answer:**')
    st.write(st.session_state.neurorag_answer)

  with col2:
    st.subheader(f'🤖 {st.session_state.competitor_model}')
    st.markdown('**Answer:**')
    st.write(st.session_state.competitor_answer)

  vote_cols = st.columns(4)

  with vote_cols[0]:
    if st.button('👈 Left is Better', type='primary', use_container_width=True):
      save_comparison(
        st.session_state.current_question,
        st.session_state.neurorag_answer,
        st.session_state.competitor_answer,
        st.session_state.competitor_model,
        'Left is Better',
      )
      st.success('Vote recorded! Left (NeuroRAG) is better.')
      # Clear current question
      del st.session_state.current_question
      st.rerun()

  with vote_cols[1]:
    if st.button('🤝 Tie', type='secondary', use_container_width=True):
      save_comparison(
        st.session_state.current_question,
        st.session_state.neurorag_answer,
        st.session_state.competitor_answer,
        st.session_state.competitor_model,
        'Tie',
      )
      st.success("Vote recorded! It's a tie.")
      del st.session_state.current_question
      st.rerun()

  with vote_cols[2]:
    if st.button('😞 Both are Bad', type='secondary', use_container_width=True):
      save_comparison(
        st.session_state.current_question,
        st.session_state.neurorag_answer,
        st.session_state.competitor_answer,
        st.session_state.competitor_model,
        'Both are Bad',
      )
      st.success('Vote recorded! Both answers are bad.')
      del st.session_state.current_question
      st.rerun()

  with vote_cols[3]:
    if st.button('👉 Right is Better', type='primary', use_container_width=True):
      save_comparison(
        st.session_state.current_question,
        st.session_state.neurorag_answer,
        st.session_state.competitor_answer,
        st.session_state.competitor_model,
        'Right is Better',
      )
      st.success('Vote recorded! Right (competitor) is better.')
      del st.session_state.current_question
      st.rerun()

# Instructions
if not hasattr(st.session_state, 'current_question'):
  st.markdown('---')
  st.markdown("""
    ### How to use:
    1. **Select a competitor model** from the sidebar
    2. **Type your question** in the text area above
    3. **Click "Generate Answers"** to get responses from both models
    4. **Vote** on which answer you think is better
    5. **Export results** to save your comparison history

    ### About NeuroRAG:
    NeuroRAG is a specialized retrieval-augmented generation system designed for neuroscience and biology questions.
    It uses multiple specialized sources and advanced reasoning techniques to provide accurate, well-grounded answers.
    """)
