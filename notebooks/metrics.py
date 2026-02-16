import string
from typing import Optional

import nltk
import numpy as np
from bert_score import score as bert_score
from dotenv import load_dotenv
from FactScoreLite import FactScore
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from summac.model_summac import SummaCConv, SummaCZS
from unidecode import unidecode

from neurorag.models.OpenRouterEmbeddings import OpenRouterEmbeddings

load_dotenv()

dict_ids: list[str] = [
  'punkt_tab',
  'punkt',
  'stopwords',
  'wordnet',
]

for dict_id in dict_ids:
  nltk.download(dict_id, quiet=True)

lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess(corpus: str) -> str:
  corpus = corpus.lower()
  stopset = (
    nltk.corpus.stopwords.words('english')
    + nltk.corpus.stopwords.words('russian')
    + list(string.punctuation)
  )
  tokens = nltk.word_tokenize(corpus)
  tokens = [t for t in tokens if t not in stopset]
  tokens = [lemmatizer.lemmatize(t) for t in tokens]
  corpus = ' '.join(tokens)
  corpus = unidecode(corpus)
  return corpus


embeddings = OpenRouterEmbeddings(model='openai/text-embedding-3-small')


def embeddings_cosine_sim_metric(
  expected_answers: list[str],
  predicted_answers: list[str],
) -> float:
  results = []

  for expected_answer, predicted_answer in zip(expected_answers, predicted_answers):
    expected_answer = preprocess(expected_answer)
    predicted_answer = preprocess(predicted_answer)

    expected_embedding = np.array(embeddings.embed_query(expected_answer))
    predicted_embedding = np.array(embeddings.embed_query(predicted_answer))

    sim = cosine_similarity(
      expected_embedding.reshape(1, -1),
      predicted_embedding.reshape(1, -1),
    )[0][0]

    results.append(sim)

  return np.mean(results)


smoothie_f = nltk.translate.bleu_score.SmoothingFunction().method4


def bleu_metric(expected_answers, predicted_answers) -> float:
  scores = []

  for expected_answer, predicted_answer in zip(expected_answers, predicted_answers):
    expected_answer = preprocess(expected_answer)
    predicted_answer = preprocess(predicted_answer)

    predicted_tokens = nltk.word_tokenize(predicted_answer)
    expected_tokens = [nltk.word_tokenize(expected_answer)]

    score = nltk.translate.bleu_score.sentence_bleu(
      expected_tokens,
      predicted_tokens,
      smoothing_function=smoothie_f,
    )

    scores.append(score)

  return np.mean(scores)


rogue_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def rogue_l_metric(expected_answers, predicted_answers) -> float:
  scores = []

  for expected_answer, predicted_answer in zip(expected_answers, predicted_answers):
    expected_answer = preprocess(expected_answer)
    predicted_answer = preprocess(predicted_answer)

    result = rogue_l_scorer.score(expected_answer, predicted_answer)

    scores.append(result['rougeL'])

  return np.mean(scores)


rogue_1_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


def rogue_1_metric(expected_answers, predicted_answers) -> float:
  scores = []

  for expected_answer, predicted_answer in zip(expected_answers, predicted_answers):
    expected_answer = preprocess(expected_answer)
    predicted_answer = preprocess(predicted_answer)

    result = rogue_1_scorer.score(expected_answer, predicted_answer)

    scores.append(result['rouge1'])

  return np.mean(scores)


def factscore_metric(expected_answers, predicted_answers) -> float:
  try:
    fact_scorer = FactScore()
    scores = []

    for expected_answer, predicted_answer in zip(expected_answers, predicted_answers):
      try:
        score, _ = fact_scorer.get_factscore(
          generations=[predicted_answer], knowledge_sources=[expected_answer]
        )
        scores.append(score)
      except Exception as e:
        print(f'Error computing FActScore for pair: {e}')
        scores.append(0.0)

    return np.mean(scores) if scores else 0.0
  except Exception as e:
    print(f'Error initializing FActScore: {e}')
    return 0.0


summac_zs: Optional[SummaCZS] = None


def summac_zs_metric(expected_answers, predicted_answers) -> float:
  global summac_zs

  scores = []

  for expected_answer, predicted_answer in zip(expected_answers, predicted_answers):
    if summac_zs is None:
      summac_zs = SummaCZS(granularity='sentence', model_name='vitc', device='cpu')

    result = summac_zs.score([expected_answer], [predicted_answer])
    scores.append(result['scores'][0])

  return np.mean(scores) if scores else 0.0


summac_conv: Optional[SummaCConv] = None


def summac_conv_metric(expected_answers, predicted_answers) -> float:
  global summac_conv

  scores = []

  for expected_answer, predicted_answer in zip(expected_answers, predicted_answers):
    if summac_conv is None:
      summac_conv = SummaCConv(
        models=['vitc'],
        bins='percentile',
        granularity='sentence',
        nli_labels='e',
        device='cpu',
        start_file='default',
        agg='mean',
      )

    result = summac_conv.score([expected_answer], [predicted_answer])
    scores.append(result['scores'][0])

  return np.mean(scores) if scores else 0.0


def bert_score_metric(
  expected_answers: list[str],
  predicted_answers: list[str],
) -> float:
  try:
    P, R, F1 = bert_score(
      predicted_answers,
      expected_answers,
      model_type='microsoft/deberta-xlarge-mnli',
      lang='en',
      rescale_with_baseline=True,
      verbose=False,
    )

    return float(F1.mean().item())
  except Exception as e:
    print(f'Error computing BERT-Score: {e}')
    return 0.0
