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


def export_results() -> str:
  """Export comparison results to JSON string"""
  if st.session_state.comparison_history:
    return json.dumps(st.session_state.comparison_history, indent=2)
  return ''


st.set_page_config(page_title='NeuroRAG LLM Arena', page_icon='🏟️', layout='wide')

st.title('🏟️ NeuroRAG LLM Arena')
st.markdown("""
Compare NeuroRAG's performance against other AI models. Ask a question and see how both models respond, then vote on which answer is better.
""")

with st.sidebar:
  st.header('⚙️ Settings')

  st.subheader('🤖 Competitor Models')
  st.markdown('Select up to 3 models to compare with NeuroRAG:')

  selected_models = st.multiselect(
    'Choose competitor models:',
    options=list(OPENROUTER_MODELS.keys()),
    default=[list(OPENROUTER_MODELS.keys())[0]],
    max_selections=3,
  )

  if not selected_models:
    st.warning('Please select at least one competitor model')

  if st.session_state.comparison_history:
    json_data = export_results()
    st.download_button(
      label='📊 Download Results',
      data=json_data,
      file_name='llm_arena_results.json',
      mime='application/json',
      type='secondary',
    )
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

if st.button(
  'Generate Answers',
  type='primary',
  disabled=not question.strip() or not selected_models,
):
  if question.strip() and selected_models:
    with st.spinner('Generating answers...'):
      neurorag_answer = get_neurorag_answer(question)

      # Generate answers for all selected competitor models
      competitor_answers = {}
      for model_name in selected_models:
        competitor_answers[model_name] = get_competitor_answer(
          question, OPENROUTER_MODELS[model_name]
        )

      st.session_state.current_question = question
      st.session_state.neurorag_answer = neurorag_answer
      st.session_state.competitor_answers = competitor_answers
      st.session_state.selected_models = selected_models

if hasattr(st.session_state, 'current_question') and st.session_state.current_question:
  # Create columns: 1 for NeuroRAG + number of competitor models
  num_competitors = len(st.session_state.selected_models)
  cols = st.columns(1 + num_competitors)

  with cols[0]:
    st.subheader('🧠 NeuroRAG')
    st.markdown('**Answer:**')
    st.write(st.session_state.neurorag_answer)

  # Display competitor models in remaining columns
  for i, model_name in enumerate(st.session_state.selected_models):
    with cols[i + 1]:
      st.subheader(f'🤖 {model_name}')
      st.markdown('**Answer:**')
      st.write(st.session_state.competitor_answers[model_name])

  # Create voting section
  st.markdown('---')
  st.subheader('🗳️ Vote on the Best Answer')

  # Create voting options
  vote_options = (
    ['NeuroRAG is Best']
    + [f'{model} is Best' for model in st.session_state.selected_models]
    + ['Tie', 'All are Bad']
  )
  num_vote_cols = len(vote_options)
  vote_cols = st.columns(num_vote_cols)

  for i, option in enumerate(vote_options):
    with vote_cols[i]:
      if st.button(
        option,
        type='secondary',
        use_container_width=True,
      ):
        # Determine which model was voted as best
        if option == 'NeuroRAG is Best':
          best_model = 'NeuroRAG'
        elif option == 'Tie':
          best_model = 'Tie'
        elif option == 'All are Bad':
          best_model = 'All are Bad'
        else:
          # Extract model name from "Model is Best"
          best_model = option.replace(' is Best', '')

        # Save comparison for each competitor model
        for model_name in st.session_state.selected_models:
          save_comparison(
            st.session_state.current_question,
            st.session_state.neurorag_answer,
            st.session_state.competitor_answers[model_name],
            model_name,
            best_model,
          )

        st.success(f'Vote recorded! {option}')
        # Clear current question
        del st.session_state.current_question
        st.rerun()

# Instructions
if not hasattr(st.session_state, 'current_question'):
  st.markdown('---')
  st.markdown("""
    ### How to use:
    1. **Select up to 3 competitor models** from the sidebar
    2. **Type your question** in the text area above
    3. **Click "Generate Answers"** to get responses from all models
    4. **Vote** on which answer you think is best
    5. **Export results** to save your comparison history

    ### About NeuroRAG:
    NeuroRAG is a specialized retrieval-augmented generation system designed for neuroscience and biology questions.
    It uses multiple specialized sources and advanced reasoning techniques to provide accurate, well-grounded answers.
    """)
