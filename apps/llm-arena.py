import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
sys.path.append('../notebooks')
sys.path.append('../neurorag')
sys.path.append('../neurorag/chains')
sys.path.append(project_root)

import streamlit as st
import warnings
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Optional
import sys

from notebooks.metrics import (
  embeddings_cosine_sim_metric,
  bleu_metric,
  rogue_l_metric,
  rogue_1_metric,
)

from neurorag.neurorag import NeuroRAG
from neurorag.models.OpenRouter import OpenRouter

warnings.filterwarnings('ignore')
load_dotenv()

if 'comparison_history' not in st.session_state:
  st.session_state.comparison_history = []

# Available OpenRouter models
OPENROUTER_MODELS = {
  'Claude 4.5 Sonnet': 'anthropic/claude-sonnet-4.5',
  'GPT-5': 'openai/gpt-5',
  'GPT-5 Nano': 'openai/gpt-5-nano',
  'Claude 3.5 Haiku': 'anthropic/claude-3.5-haiku',
  'Mistral Large': 'mistralai/mistral-large',
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
  neurorag = NeuroRAG(temperature=0, debug=True)
  neurorag.compile()
  result = neurorag.invoke(question)
  return result.get('generation', 'No answer generated')


def get_competitor_answer(question: str, model_name: str) -> str:
  """Get answer from competitor model"""
  llm = get_openrouter_llm(model_name)
  if llm:
    return llm.invoke(question)
  return 'Error: Could not initialize model'


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


def get_available_datasets() -> list[str]:
  """Get list of available CSV datasets"""
  datasets_dir = '../datasets'
  csv_files = []
  if os.path.exists(datasets_dir):
    for file in os.listdir(datasets_dir):
      if file.endswith('.csv'):
        csv_files.append(file)
  return csv_files


def load_dataset(dataset_path: str) -> tuple[list[str], list[str], Optional[list[str]]]:
  """Load dataset and return questions, answers, and optional categories"""
  try:
    df = pd.read_csv(dataset_path)
    if 'question' in df.columns and 'answer' in df.columns:
      questions = df['question'].dropna().tolist()
      answers = df['answer'].dropna().tolist()
      categories = None
      if 'category' in df.columns:
        categories = df['category'].fillna('Unknown').tolist()
      return questions, answers, categories
    else:
      st.error(
        f"Dataset must have 'question' and 'answer' columns. Found columns: {list(df.columns)}"
      )
      return [], [], None
  except Exception as e:
    st.error(f'Error loading dataset: {e}')
    return [], [], None


def build_masked_names(model_list):
  return {name: f'Model {i + 1}' for i, name in enumerate(model_list)}


MASKED_NAMES = build_masked_names(list(OPENROUTER_MODELS.keys()) + ['NeuroRAG'])


def evaluate_models_on_dataset(
  questions: list[str],
  expected_answers: list[str],
  selected_models: list[str],
  categories: Optional[list[str]] = None,
) -> dict:
  """Evaluate all models on the dataset and return metrics"""
  results: dict[str, Any] = {
    'dataset_name': 'uploaded_dataset',
    'questions': questions,
    'expected_answers': expected_answers,
    'categories': categories,
    'neurorag_answers': [],
    'competitor_answers': {},
    'metrics': {},
    'metrics_by_category': {},
  }

  # Initialize competitor answers dict
  for model_name in selected_models:
    results['competitor_answers'][model_name] = []

  # Generate answers for all questions
  with st.spinner('Generating NeuroRAG answers...'):
    for question in questions:
      answer = get_neurorag_answer(question)
      results['neurorag_answers'].append(answer)

  # Generate competitor answers
  for model_name in selected_models:
    with st.spinner(f'Generating {model_name} answers...'):
      for question in questions:
        answer = get_competitor_answer(question, OPENROUTER_MODELS[model_name])
        results['competitor_answers'][model_name].append(answer)

  # Calculate metrics for each model
  with st.spinner('Calculating metrics...'):
    # NeuroRAG metrics
    try:
      neurorag_metrics = {
        'cosine_similarity': embeddings_cosine_sim_metric(
          expected_answers, results['neurorag_answers']
        ),
        'bleu': bleu_metric(expected_answers, results['neurorag_answers']),
        'rouge_l': rogue_l_metric(expected_answers, results['neurorag_answers']),
        'rouge_1': rogue_1_metric(expected_answers, results['neurorag_answers']),
      }
      results['metrics'][MASKED_NAMES['NeuroRAG']] = neurorag_metrics
    except Exception as e:
      st.error(f'Error calculating NeuroRAG metrics: {e}')
      results['metrics']['NeuroRAG'] = {'error': str(e)}

    # Competitor models metrics
    for model_name in selected_models:
      try:
        competitor_metrics = {
          'cosine_similarity': embeddings_cosine_sim_metric(
            expected_answers, results['competitor_answers'][model_name]
          ),
          'bleu': bleu_metric(
            expected_answers, results['competitor_answers'][model_name]
          ),
          'rouge_l': rogue_l_metric(
            expected_answers, results['competitor_answers'][model_name]
          ),
          'rouge_1': rogue_1_metric(
            expected_answers, results['competitor_answers'][model_name]
          ),
        }
        results['metrics'][model_name] = competitor_metrics
      except Exception as e:
        st.error(f'Error calculating {model_name} metrics: {e}')
        results['metrics'][model_name] = {'error': str(e)}

  # Calculate metrics by category if categories are available
  if categories:
    with st.spinner('Calculating metrics by category...'):
      unique_categories = list(set(categories))

      for category in unique_categories:
        # Get indices for this category
        category_indices = [i for i, cat in enumerate(categories) if cat == category]

        if not category_indices:
          continue

        # Extract data for this category
        cat_expected = [expected_answers[i] for i in category_indices]
        cat_neurorag = [results['neurorag_answers'][i] for i in category_indices]

        category_metrics = {}

        # NeuroRAG metrics for this category
        try:
          category_metrics['NeuroRAG'] = {
            'cosine_similarity': embeddings_cosine_sim_metric(
              cat_expected, cat_neurorag
            ),
            'bleu': bleu_metric(cat_expected, cat_neurorag),
            'rouge_l': rogue_l_metric(cat_expected, cat_neurorag),
            'rouge_1': rogue_1_metric(cat_expected, cat_neurorag),
          }
        except Exception as e:
          category_metrics['NeuroRAG'] = {'error': str(e)}

        # Competitor models metrics for this category
        for model_name in selected_models:
          try:
            cat_competitor = [
              results['competitor_answers'][model_name][i] for i in category_indices
            ]
            category_metrics[model_name] = {
              'cosine_similarity': embeddings_cosine_sim_metric(
                cat_expected, cat_competitor
              ),
              'bleu': bleu_metric(cat_expected, cat_competitor),
              'rouge_l': rogue_l_metric(cat_expected, cat_competitor),
              'rouge_1': rogue_1_metric(cat_expected, cat_competitor),
            }
          except Exception as e:
            category_metrics[model_name] = {'error': str(e)}

        results['metrics_by_category'][category] = {
          'count': len(category_indices),
          'metrics': category_metrics,
        }

  return results


st.set_page_config(page_title='NeuroRAG LLM Arena', page_icon='🏟️', layout='wide')

st.title('🏟️ NeuroRAG LLM Arena')
st.markdown("""
Compare NeuroRAG's performance against other AI models. Ask a question and see how both models respond, then vote on which answer is better.
""")

with st.sidebar:
  st.header('⚙️ Settings')

  st.subheader('🤖 Competitor Models')
  st.markdown('Select up to 3 models to compare with NeuroRAG:')

  real_model_names = list(OPENROUTER_MODELS.keys())

  selected_models = st.multiselect(
    'Choose competitor models:',
    options=real_model_names,
    default=[real_model_names[0]],
    max_selections=3,
  )

  if not selected_models:
    st.warning('Please select at least one competitor model')

  selected_models_masked = [MASKED_NAMES[m] for m in selected_models]

  st.session_state.selected_models_masked = selected_models_masked

  # Dataset evaluation section
  st.markdown('---')
  st.subheader('📊 Dataset Evaluation')

  # Available datasets
  available_datasets = get_available_datasets()
  if available_datasets:
    st.markdown('**Available datasets:**')
    selected_dataset = st.selectbox(
      'Choose a dataset:', options=[''] + available_datasets, index=0
    )

    if selected_dataset:
      dataset_path = f'../datasets/{selected_dataset}'
      questions, answers, categories = load_dataset(dataset_path)

      if questions and answers:
        category_info = (
          f' (with {len(set(categories))} categories)' if categories else ''
        )
        st.success(
          f'Loaded {len(questions)} questions from {selected_dataset}{category_info}'
        )

        if st.button('🚀 Run Evaluation', type='primary'):
          if len(questions) > 10:
            st.warning('Dataset is large. This may take a while...')

          evaluation_results = evaluate_models_on_dataset(
            questions, answers, selected_models, categories
          )
          st.session_state.evaluation_results = evaluation_results
          st.success('Evaluation completed!')
          st.rerun()

  # File upload
  st.markdown('**Or upload your own dataset:**')
  uploaded_file = st.file_uploader(
    'Upload CSV file with question and answer columns',
    type=['csv'],
    help='CSV must have columns named "question" and "answer". Optional "category" column for category-based metrics.',
  )

  if uploaded_file is not None:
    try:
      df = pd.read_csv(uploaded_file)
      if 'question' in df.columns and 'answer' in df.columns:
        questions = df['question'].dropna().tolist()
        answers = df['answer'].dropna().tolist()
        categories = None
        if 'category' in df.columns:
          categories = df['category'].fillna('Unknown').tolist()

        category_info = (
          f' (with {len(set(categories))} categories)' if categories else ''
        )
        st.success(f'Uploaded dataset with {len(questions)} questions{category_info}')

        if st.button('🚀 Evaluate Uploaded Dataset', type='primary'):
          if len(questions) > 10:
            st.warning('Dataset is large. This may take a while...')

          evaluation_results = evaluate_models_on_dataset(
            questions, answers, selected_models, categories
          )
          evaluation_results['dataset_name'] = uploaded_file.name
          st.session_state.evaluation_results = evaluation_results
          st.success('Evaluation completed!')
          st.rerun()
      else:
        st.error(
          f"Uploaded file must have 'question' and 'answer' columns. Found: {list(df.columns)}"
        )
    except Exception as e:
      st.error(f'Error reading uploaded file: {e}')

  # Export comparison results
  st.markdown('---')
  st.subheader('📤 Export Results')

  if st.session_state.comparison_history:
    json_data = export_results()
    st.download_button(
      label='📊 Download Comparison Results',
      data=json_data,
      file_name='llm_arena_results.json',
      mime='application/json',
      type='secondary',
    )
  else:
    st.warning('No comparison results to export')

  # Show comparison history
  if st.session_state.comparison_history:
    st.header('📝 History')
    for i, comp in enumerate(st.session_state.comparison_history[-5:]):  # Show last 5
      with st.expander(f'Q{i + 1}: {comp["question"][:50]}...'):
        st.write(f'**Model:** {MASKED_NAMES.get(comp["competitor_model"], "Model")}')
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
    st.subheader(f'🧠 {MASKED_NAMES["NeuroRAG"]}')
    st.markdown('**Answer:**')
    st.write(st.session_state.neurorag_answer)

  # Display competitor models in remaining columns
  for i, model_name in enumerate(st.session_state.selected_models):
    masked = MASKED_NAMES[model_name]
    with cols[i + 1]:
      st.subheader(f'🤖 {masked}')
      st.markdown('**Answer:**')
      st.write(st.session_state.competitor_answers[model_name])

  st.markdown('---')
  st.subheader('🗳️ Vote on the Best Answer')

  vote_options = (
    [f'{MASKED_NAMES["NeuroRAG"]} is Best']
    + [f'{MASKED_NAMES[model]} is Best' for model in st.session_state.selected_models]
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
        if option == f'{MASKED_NAMES["NeuroRAG"]} is Best':
          best_model = MASKED_NAMES['NeuroRAG']
        elif option == 'Tie':
          best_model = 'Tie'
        elif option == 'All are Bad':
          best_model = 'All are Bad'
        else:
          best_model = option.replace(' is Best', '')

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
    2. **Choose a dataset** or upload your own CSV file
    3. **Run evaluation** to test all models on the dataset
    4. **Type individual questions** to compare models manually
    5. **Vote** on which answer you think is best
    6. **Export results** to save your comparison history and evaluation results

    ### About NeuroRAG:
    NeuroRAG is a specialized retrieval-augmented generation system designed for neuroscience and biology questions.
    It uses multiple specialized sources and advanced reasoning techniques to provide accurate, well-grounded answers.

    ### Metrics Explained:
    - **Cosine Similarity**: Semantic similarity between expected and generated answers
    - **BLEU**: N-gram overlap between expected and generated answers
    - **ROUGE-L/1**: Longest common subsequence and unigram overlap
    """)

# Display evaluation results if available
print(st.session_state)
if (
  'evaluation_results' in st.session_state
  and st.session_state.evaluation_results is not None
):
  st.markdown('---')
  st.header('📊 Dataset Evaluation Results')

  results = st.session_state.evaluation_results

  st.subheader(f'Dataset: {results["dataset_name"]}')
  st.markdown(f'**Total Questions:** {len(results["questions"])}')

  st.subheader('📈 Metrics Comparison')

  metrics_data = []
  for model_name, metrics in results['metrics'].items():
    if 'error' not in metrics:
      metrics_data.append(
        {
          'Model': MASKED_NAMES.get(model_name, model_name),
          'Cosine Similarity': f'{metrics["cosine_similarity"]:.4f}',
          'BLEU': f'{metrics["bleu"]:.4f}',
          'ROUGE-L': f'{metrics["rouge_l"]:.4f}',
          'ROUGE-1': f'{metrics["rouge_1"]:.4f}',
        }
      )

  if metrics_data:
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    # Display metrics by category if available
    if results.get('metrics_by_category'):
      st.subheader('📊 Metrics by Category')

      category_metrics_data = []
      for category, cat_data in results['metrics_by_category'].items():
        count = cat_data['count']
        for model_name, metrics in cat_data['metrics'].items():
          if 'error' not in metrics:
            category_metrics_data.append(
              {
                'Category': category,
                'Count': count,
                'Model': model_name,
                'Cosine Similarity': f'{metrics["cosine_similarity"]:.4f}',
                'BLEU': f'{metrics["bleu"]:.4f}',
                'ROUGE-L': f'{metrics["rouge_l"]:.4f}',
                'ROUGE-1': f'{metrics["rouge_1"]:.4f}',
              }
            )

      if category_metrics_data:
        category_metrics_df = pd.DataFrame(category_metrics_data)
        st.dataframe(category_metrics_df, use_container_width=True)

    json_data = json.dumps(results, indent=2, ensure_ascii=False)
    st.download_button(
      label='📥 Download Evaluation Results',
      data=json_data,
      file_name=f'evaluation_results_{results["dataset_name"]}.json',
      mime='application/json',
      type='primary',
    )
  else:
    st.error('No valid metrics computed')

  st.subheader('📝 Sample Questions & Answers')
  sample_size = min(5, len(results['questions']))

  for i in range(sample_size):
    with st.expander(f'Question {i + 1}: {results["questions"][i][:100]}...'):
      st.markdown('**Question:**')
      st.write(results['questions'][i])

      if results['expected_answers'][i]:
        st.markdown('**Reference Answer:**')
        st.write(results['expected_answers'][i])

      st.markdown('**NeuroRAG Answer:**')
      st.write(results['neurorag_answers'][i])

      for model_name in results['competitor_answers']:
        st.markdown(f'**{model_name} Answer:**')
        st.write(results['competitor_answers'][model_name][i])

  if st.button('🗑️ Clear Results', type='secondary'):
    del st.session_state.evaluation_results
    st.rerun()
