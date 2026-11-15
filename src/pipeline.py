"""
End to End investigation assistant pipeline.

User query > Query Understanding > Vectorless Retrieval > context > LLM > Answer
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from src.llm import build_context_blocks, generate_answer
from src.retriever import QueryFilters, RetrievalResult, retrieve
from src.utils import RateLimiter, extract_claim_ids, logger

load_dotenv()

@dataclass
class PipelineResponse:
  answer: str
  retrieval: RetrievalResult
  citations: list[dict[str, Any]] = field(default_factory=list)
  llm_latency_seconds: float =0.0
  total_latency_seconds: float = 0.0
  resolved_query: str = ""
  error: str | None = None

_rate_limiter = RateLimiter(
  max_calls= int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")),
  period_seconds= 60.0,
)

def resolve_followups( query: str, chat_history: list[dict[str, str]]) -> str:
  """
  Resolve simple follow-up questions using a Claim ID form recent history.
  
  If the user says "it", "that claim", etc., the function adds the most
  recently mentioned claim id to the query.
  
  Example:
    previous: "Tell me about CLM-1023"
    Current: "Why was it referred to SIU?"
    > "Why was it referred to SIU? (referring to CLM-1023)
  """


  lower = query.lower()

  # Look for phrases that indicate the user is referring to an earlier claim.
  pronouns = (
    "why wat it",
    "what about it",
    "that claim",
    "this claim",
    "the claim",
  )

  has_pronoun = any(p in lower for p in pronouns) or (
    re.search(r"\bit\b", lower) and not extract_claim_ids(query)
  )

  #Nothing to resolve, so return the original query.
  if not has_pronoun:
    return query
  # Search recent conversation turns for the most recent Claim ID.
  for turn in reversed(chat_history):
    ids = extract_claim_ids(turn.get("content", ""))

    if ids:
      resolved =f"{query} (referring to {ids[0]})"
      logger.info("Resolved follow-up query to include %s", ids[0])
      return resolved

  # No previous Claim ID was found.
  return query

def run_pipeline(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    top_k: int=5,
    sidebar_filters: dict[str, Any] | None = None,
    skip_llm: bool = False,
) -> PipelineResponse:
  """
  Run the complete chatbot pipeline from query to final answer.
  
  It handles query validation, follow_up  resolution, retrieval,
  rate limiting, LLM generation, errors, citations, and latency tracking.
  
  Example:
  User question

  Resolve follow-up

  Retrieve evidence

  Generated LLM answer

  Return answer + sources = metrics
  
  """

  total_start = time.perf_counter()
  chat_history =chat_history or []

  # Clean and validate the user's query
  query =(query or "").strip()

  if not query:
    # Return a friendly response instead of running the pipeline.
    empty = RetrievalResult(
      documents=[],
      filters=QueryFilters(),
      exact_count = 0,
      bm25_count =0,
      filtered_count= 0,
      latency_seconds=0.0,
    )

    return PipelineResponse(
      answer="Please enter a question about a claim, policy or investigation topic.",
      retrieval=empty,
      error="empty_query",
      total_latency_seconds=0.0,
    )

  # Resolved references such as "that claim" using conversation history.
  resolved = resolve_followups(query, chat_history)

  try:
    # run the vectorless retrieval pipeline.
    retrieval = retrieve(
      resolved,
      top_k=top_k,
      sidebar_filters=sidebar_filters,
    )

  except Exception as exc:  # noqa: BLE001 - retrieval adapters can raise diverse runtime errors
    # Return a structured error if retrieval fails.
    empty = RetrievalResult(
      documents=[],
      filters=QueryFilters(free_text=resolved),
      exact_count=0,
      bm25_count=0,
      filtered_count=0,
      latency_seconds=0.0,

    )

    return PipelineResponse(
      answer=f"Retrieval error:{exc}",
      retrieval=empty,
      error=str(exc),
      resolved_query=resolved,
      total_latency_seconds=time.perf_counter()-total_start,
    )

  # Prepare citation information from the retrieved documents.
  _, citations= build_context_blocks(retrieval.documents)

  #useful for testing retrieval without making an LLM API call.

  if skip_llm:
    return PipelineResponse(
      answer="",
      retrieval=retrieval,
      citations=citations,
      resolved_query=resolved,
      total_latency_seconds=time.perf_counter() - total_start,

    )
  # Check the application level LLM rate limit before making the api call.
  if not _rate_limiter.allow():
    return PipelineResponse(
      answer=(
        "Rate limit reached. Please wait a moment before sending another"
        "question. (Teaching demo: in-memory limit of "
        f"{_rate_limiter.max_calls} LLM calls per minute.)"
      ),
      retrieval=retrieval,
      citations=citations,
      resolved_query=resolved,
      error="rate_limited",
      total_latency_seconds=time.perf_counter() - total_start,

    )

  llm_start=time.perf_counter()

  try:
    #generate the final answer using the retrieved evidence.

    answer = generate_answer(
      resolved,
      retrieval.documents,
      chat_history=chat_history,
    )

    llm_latency = time.perf_counter() - llm_start
  except Exception as  exc:
    llm_latency = time.perf_counter() - llm_start
    logger.error("LLM error: %s", exc)

    # Even if the LLM fails, return the retrieved evidence to the user.

    fallback =(
      "I retrieved relevant evidence, but the language model is unavailable"
      f"right now ({exc}).\n\n"
      "Retrieved claim IDs / sources:\n"
      +"\n".join(
        f"-{c.get('source')} | {c.get('claim_id') or 'n/a'}"
        for c in citations
      )
    )

    return PipelineResponse(
      answer=fallback,
      retrieval=retrieval,
      citations=citations,
      llm_latency_seconds=llm_latency,
      resolved_query=resolved,
      error=str(exc),
      total_latency_seconds=time.perf_counter() - total_start,
  )
  
  total_latency = time.perf_counter() - total_start

  # Log retrieval and LLM performance separately.
  logger.info(
    "Pipeline complete | retrieval=%.3fs llm=%.3fs total=%.3fs",
    retrieval.latency_seconds,
    llm_latency,
    total_latency,
  )
  return PipelineResponse(
    answer=answer,
    retrieval=retrieval,
    citations=citations,
    llm_latency_seconds=llm_latency,
    total_latency_seconds=total_latency,
    resolved_query=resolved,
  )  