"""FLARE (Forward-Looking Active REtrieval augmented generation).

Generates answer sentence-by-sentence. After each sentence, the LLM checks
whether it contains uncertain claims. If so, those claims are used as
retrieval queries to fetch additional evidence, and the sentence is
regenerated with enriched context.

Reference: Jiang et al., "Active Retrieval Augmented Generation" (2023).
"""

import asyncio
from typing import Callable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from neurorag.chains.json_extractor import JsonExtractor


# ── Prompts ──────────────────────────────────────────────────────────────────

GENERATE_NEXT_PROMPT = PromptTemplate.from_template(
  """You are a scientific assistant. Continue writing the answer to the QUERY.

### Rules
- Write EXACTLY ONE next sentence that logically follows PARTIAL_ANSWER.
- Base your sentence on the CONTEXT provided.
- If the answer is already complete, respond with exactly: [DONE]
- Do NOT repeat information already in PARTIAL_ANSWER.

QUERY:
{query}

CONTEXT:
{context}

PARTIAL_ANSWER (so far):
{partial_answer}

Next sentence:"""
)

IDENTIFY_UNCERTAIN_PROMPT = PromptTemplate.from_template(
  """Analyze the following SENTENCE written in response to QUERY.
Identify claims that are specific, factual, and may need verification
(e.g., numbers, mechanisms, named entities, causal claims).

If ALL claims are clearly supported by CONTEXT, respond with exactly: []
Otherwise, return a JSON array of short search queries that would help
verify the uncertain claims. Maximum 2 queries.

QUERY:
{query}

CONTEXT:
{context}

SENTENCE:
{sentence}

Respond with ONLY a JSON array (e.g., ["query 1", "query 2"] or []):"""
)

REGENERATE_PROMPT = PromptTemplate.from_template(
  """Rewrite the SENTENCE below to be more accurate, using the ADDITIONAL_EVIDENCE.
Keep it as one sentence. Preserve the meaning but correct any inaccuracies.
If the sentence is already accurate, return it unchanged.

QUERY:
{query}

ORIGINAL_CONTEXT:
{context}

ADDITIONAL_EVIDENCE:
{evidence}

SENTENCE:
{sentence}

Rewritten sentence:"""
)


class FlareChain:
  """FLARE: iterative sentence-level generation with active retrieval."""

  def __init__(
    self,
    llm,
    retriever_fn: Callable[[str], list[Document]],
    max_sentences: int = 15,
    max_retrievals: int = 6,
  ):
    """
    Args:
      llm: LangChain-compatible LLM.
      retriever_fn: Callable that takes a query string and returns Documents.
      max_sentences: Max sentences to generate before stopping.
      max_retrievals: Max total FLARE retrieval rounds to limit latency.
    """
    self.llm = llm
    self.retriever_fn = retriever_fn
    self.max_sentences = max_sentences
    self.max_retrievals = max_retrievals

    self.generate_chain = (
      GENERATE_NEXT_PROMPT | llm | StrOutputParser()
    )
    self.identify_chain = (
      IDENTIFY_UNCERTAIN_PROMPT | llm | StrOutputParser() | JsonExtractor()
    )
    self.regenerate_chain = (
      REGENERATE_PROMPT | llm | StrOutputParser()
    )

  def _retrieve(self, queries: list[str]) -> str:
    """Retrieve documents for multiple queries and combine."""
    all_docs: list[Document] = []
    seen: set[str] = set()
    for q in queries:
      try:
        docs = self.retriever_fn(q)
        for doc in docs:
          if doc.page_content not in seen:
            seen.add(doc.page_content)
            all_docs.append(doc)
      except Exception:
        continue
    return '\n\n'.join(doc.page_content for doc in all_docs[:5])

  def invoke(self, query: str, context: str) -> str:
    """Generate an answer using FLARE.

    Args:
      query: User question.
      context: Initial retrieved context.

    Returns:
      Complete generated answer.
    """
    partial_answer = ''
    total_retrievals = 0

    for _ in range(self.max_sentences):
      next_sentence = self.generate_chain.invoke({
        'query': query,
        'context': context,
        'partial_answer': partial_answer or '(empty — start the answer)',
      }).strip()

      if '[DONE]' in next_sentence or not next_sentence:
        break

      # Clean up: take only the first sentence if the LLM over-generated
      for delim in ['. ', '.\n']:
        if delim in next_sentence:
          next_sentence = next_sentence[: next_sentence.index(delim) + 1]
          break

      if total_retrievals < self.max_retrievals:
        try:
          uncertain_queries = self.identify_chain.invoke({
            'query': query,
            'context': context,
            'sentence': next_sentence,
          })

          if isinstance(uncertain_queries, list) and len(uncertain_queries) > 0:
            evidence = self._retrieve(uncertain_queries)
            if evidence:
              next_sentence = self.regenerate_chain.invoke({
                'query': query,
                'context': context,
                'evidence': evidence,
                'sentence': next_sentence,
              }).strip()
              # Enrich the context for subsequent sentences
              context = context + '\n\n' + evidence
              total_retrievals += 1
        except Exception:
          pass

      partial_answer = (partial_answer + ' ' + next_sentence).strip()

    return partial_answer

  async def ainvoke(self, query: str, context: str) -> str:
    """Async version of invoke."""
    partial_answer = ''
    total_retrievals = 0

    generate_chain_async = GENERATE_NEXT_PROMPT | self.llm | StrOutputParser()
    identify_chain_async = (
      IDENTIFY_UNCERTAIN_PROMPT | self.llm | StrOutputParser() | JsonExtractor()
    )
    regenerate_chain_async = REGENERATE_PROMPT | self.llm | StrOutputParser()

    for _ in range(self.max_sentences):
      next_sentence = (
        await generate_chain_async.ainvoke({
          'query': query,
          'context': context,
          'partial_answer': partial_answer or '(empty — start the answer)',
        })
      ).strip()

      if '[DONE]' in next_sentence or not next_sentence:
        break

      for delim in ['. ', '.\n']:
        if delim in next_sentence:
          next_sentence = next_sentence[: next_sentence.index(delim) + 1]
          break

      if total_retrievals < self.max_retrievals:
        try:
          uncertain_queries = await identify_chain_async.ainvoke({
            'query': query,
            'context': context,
            'sentence': next_sentence,
          })

          if isinstance(uncertain_queries, list) and len(uncertain_queries) > 0:
            evidence = self._retrieve(uncertain_queries)
            if evidence:
              next_sentence = (
                await regenerate_chain_async.ainvoke({
                  'query': query,
                  'context': context,
                  'evidence': evidence,
                  'sentence': next_sentence,
                })
              ).strip()
              context = context + '\n\n' + evidence
              total_retrievals += 1
        except Exception:
          pass

      partial_answer = (partial_answer + ' ' + next_sentence).strip()

    return partial_answer
