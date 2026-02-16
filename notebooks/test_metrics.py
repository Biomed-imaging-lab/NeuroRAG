from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from notebooks.metrics import (
  _decompose_to_atomic_facts,
  _verify_fact,
  factscore_metric,
)


def _make_chat_response(content: str) -> MagicMock:
  """Helper to build a mock OpenAI chat completion response."""
  response = MagicMock()
  response.choices = [MagicMock()]
  response.choices[0].message.content = content
  return response


class TestDecomposeToAtomicFacts:
  """Tests for _decompose_to_atomic_facts."""

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_list_of_facts(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response(
      '["The sky is blue.", "Water is wet."]'
    )
    facts = _decompose_to_atomic_facts('The sky is blue and water is wet.')
    assert facts == ['The sky is blue.', 'Water is wet.']

  @patch('notebooks.metrics.openrouter_client')
  def test_extracts_json_from_surrounding_text(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response(
      'Here are the facts:\n["Fact one.", "Fact two."]\nDone.'
    )
    facts = _decompose_to_atomic_facts('Some text.')
    assert facts == ['Fact one.', 'Fact two.']

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_empty_list_on_invalid_json(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response(
      'not valid json at all'
    )
    facts = _decompose_to_atomic_facts('Some text.')
    assert facts == []

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_empty_list_on_none_content(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response(None)
    # content is None, fallback to '[]'
    response = mock_client.chat.completions.create.return_value
    response.choices[0].message.content = None
    facts = _decompose_to_atomic_facts('Some text.')
    assert facts == []

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_empty_list_on_malformed_json_array(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response(
      '["unclosed array'
    )
    facts = _decompose_to_atomic_facts('Some text.')
    assert facts == []


class TestVerifyFact:
  """Tests for _verify_fact."""

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_true_when_supported(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response('true')
    assert _verify_fact('The sky is blue.', 'The sky is blue on a clear day.') is True

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_false_when_not_supported(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response('false')
    assert _verify_fact('The sky is green.', 'The sky is blue.') is False

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_false_when_response_is_ambiguous(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response(
      'It is true that this is false.'
    )
    # Contains both "true" and "false" → should return False
    assert _verify_fact('Ambiguous fact.', 'Reference.') is False

  @patch('notebooks.metrics.openrouter_client')
  def test_returns_false_on_empty_response(self, mock_client):
    mock_client.chat.completions.create.return_value = _make_chat_response('')
    assert _verify_fact('Some fact.', 'Some reference.') is False


class TestFactscoreMetric:
  """Tests for factscore_metric."""

  @patch('notebooks.metrics._verify_fact')
  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_perfect_score_without_penalty(self, mock_decompose, mock_verify):
    # 12 facts, all supported → no length penalty (12 >= gamma=10)
    mock_decompose.return_value = [f'fact {i}' for i in range(12)]
    mock_verify.return_value = True

    score = factscore_metric(['reference'], ['prediction'], gamma=10)
    assert score == pytest.approx(1.0)

  @patch('notebooks.metrics._verify_fact')
  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_zero_score_when_nothing_supported(self, mock_decompose, mock_verify):
    mock_decompose.return_value = [f'fact {i}' for i in range(12)]
    mock_verify.return_value = False

    score = factscore_metric(['reference'], ['prediction'], gamma=10)
    assert score == pytest.approx(0.0)

  @patch('notebooks.metrics._verify_fact')
  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_partial_score(self, mock_decompose, mock_verify):
    mock_decompose.return_value = [f'fact {i}' for i in range(10)]
    # 7 out of 10 supported
    mock_verify.side_effect = [True] * 7 + [False] * 3

    score = factscore_metric(['reference'], ['prediction'], gamma=10)
    assert score == pytest.approx(0.7)

  @patch('notebooks.metrics._verify_fact')
  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_length_penalty_applied_when_few_facts(self, mock_decompose, mock_verify):
    # Only 3 facts, all supported, gamma=10 → penalty applies
    mock_decompose.return_value = ['fact 0', 'fact 1', 'fact 2']
    mock_verify.return_value = True

    score = factscore_metric(['reference'], ['prediction'], gamma=10)
    expected_penalty = np.exp(1 - 10 / 3)
    assert score == pytest.approx(expected_penalty * 1.0)
    assert score < 1.0

  @patch('notebooks.metrics._verify_fact')
  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_no_penalty_when_gamma_is_zero(self, mock_decompose, mock_verify):
    mock_decompose.return_value = ['fact 0', 'fact 1']
    mock_verify.return_value = True

    score = factscore_metric(['reference'], ['prediction'], gamma=0)
    assert score == pytest.approx(1.0)

  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_returns_zero_when_no_facts_decomposed(self, mock_decompose):
    mock_decompose.return_value = []

    score = factscore_metric(['reference'], ['prediction'])
    assert score == pytest.approx(0.0)

  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_returns_zero_on_exception(self, mock_decompose):
    mock_decompose.side_effect = RuntimeError('API error')

    score = factscore_metric(['reference'], ['prediction'])
    assert score == pytest.approx(0.0)

  def test_returns_zero_on_empty_inputs(self):
    score = factscore_metric([], [])
    assert score == pytest.approx(0.0)

  @patch('notebooks.metrics._verify_fact')
  @patch('notebooks.metrics._decompose_to_atomic_facts')
  def test_averages_across_multiple_pairs(self, mock_decompose, mock_verify):
    # Pair 1: 10 facts, all supported → 1.0
    # Pair 2: 10 facts, none supported → 0.0
    mock_decompose.return_value = [f'fact {i}' for i in range(10)]
    mock_verify.side_effect = [True] * 10 + [False] * 10

    score = factscore_metric(
      ['ref1', 'ref2'],
      ['pred1', 'pred2'],
      gamma=10,
    )
    assert score == pytest.approx(0.5)
