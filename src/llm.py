"""
Simple Openai LLM client with retry, timeout, and context limits.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv  # cspell:ignore dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from src.utils import logger, retry_with_backoff, truncate_context

load_dotenv()

SYSTEM_PROMPT = """ You are an Insurance claims investigation Assistant for internal adjusters and SIU analysts.

Rules:
1. Answer ONLY using the retrieved evidence provided in the user message.
2.Do no invent claim facts, amounts, dates, or referral reasons.
3. If evidence is insufficient, say so clearly.
4. Mention relevant claim IDs when they appear in the evidence.
5. keep answers concise and investigation-oriented.
6. Distinguish facts from conclusions (e.g., "the note states..." vs "this suggests...").
7. End with a sources section listing source file names and claim IDs / page numbers when available.

Format:
Answer:
<your Name>

Sources:
-<sources> (Claim ID / page if available)
"""

def get_client() -> OpenAI:
  api_key = os.getenv("OPENAI_API_KEY", "").strip()
  if not api_key or api_key.startswith("your_"):
    raise ValueError(
      "OPENAI_API_KEY is not configured. Copy .env. example to .env and set your key."
    )
  timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
  return OpenAI(api_key=api_key, timeout=timeout)

def build_context_blocks(
    scored_docs: list[Any], max_chars: int | None = None
)-> tuple[str, list[dict[str, Any]]]:
  """
  Build the context sent to the LLM and the citation information for each document.
  
  Each retrieved document gets a header containing its source, claim ID,
  retrieval method and scores. the final context is limited to Max_chars.
  
  Example:
      Retrieved documents
      Context blocks + citation metadata
      LLM Context + ui citation
  """

  if max_chars is None:
    #use the configured context limit, or 12,000 characters by default.
    max_chars = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

  blocks: list[str] =[]
  citations: list[dict[str, Any]] = []

  for i, item in enumerate(scored_docs, start=1):
    # Support both scored-result objects and plain documents dictionaries.
    doc = item.document if hasattr(item, "document") else item
    meta = doc.get("metadata") or {}

    # Build a readable header containing useful retrieval information
    header_parts = [f"[{i}] Source: {doc.get('source', 'unknown')}"]

    if meta.get("claim_id"):
      header_parts.append(f"Claim ID: {meta['claim_id']}")
    if meta.get("page"):
      header_parts.append(f"Page: {meta['page']}")
    if meta.get("section"):
      header_parts.append(f"Section: {meta['section']}")

    methods = getattr(item, "methods", None)
    if methods:
      header_parts.append(f"Retrieval: {','.join(methods)}")

    bm25 = getattr(item, "bm25_score", None)
    if bm25 is not None:
      header_parts.append(f"BM25: {bm25: .3f}")

    block = "|".join(header_parts) + "\n" + doc.get("text","")
    blocks.append(block)

    # Score citation details separated so the ui can display them.

    citations.append(
      {
        "source": doc.get("source", "unknown"),
        "claim_id": meta.get("claim_id", ""),
        "page": meta.get("page"),
        "section": meta.get("section"),
        "methods": list(methods or []),
        "bm25_score": bm25,
        "score": getattr(item, "score", None),
        "snippet": doc.get("text", "")[:240],
      }
    )
  # LImit the context before sending it to the LLM.
  truncated = truncate_context(blocks, max_chars)

  # Keep citations aligned with the documents included in the context.
  citations = citations[: len(truncated)]

  return "\n\n---\n\n".join(truncated), citations

def generate_answer(
    query: str,
    scored_docs: list[Any],
    chat_history: list[dict[str, str]] | None = None,    
)-> str:
  """
  Generate an answer using only the documents retrieved by the search system.
  
  The function combines retrieved evidence with recent conversation history and sends it to the LLM. If no relevant documents are found, it avoids generating an unsupported answer.
  
  Example:
  Query > "Why was CLM-1023 referred to SUI?"
  Retrieved documents > Claim notes + investigation report.
  Output > Evidence-based answer with sources.
  """

  # if retrieval found nothing useful, do not ask the LLM to guess an answer.

  if not scored_docs:
    return(
      "Answer: \n I could not find enough evidence in the claims Knowledge base." \
      "to answer this question. Try a more specific claim ID, status, state,"
      "or loss type.\n\n Sources:\n- None"
    )

  # Context retrieved documents into the context that will be given to the LLM.

  context, _= build_context_blocks(scored_docs)

  history_lines: list[str] = []

  # Include only the most recent conversation turns to keep the prompt manageable.
  for turn in (chat_history or [])[-6:]:
    role = turn.get("role", "user")
    content = turn.get("content", "")
    history_lines.append(f"{role.upper()}: {content}")
  history_block ="\n".join(history_lines) if history_lines else "(none)"


  # Build the final prompt containing conversation, question, and retrieved evidence.

  user_prompt =(
    f"Conversation history: \n{history_block}\n\n"
    f"Current question:\n{query}\n\n"
    f"Retrieved evidence:\n{context}\n\n"
    "Using only the evidence above, answer the current question."
  )

  # Get the configured LLM client and model.
  client = get_client()
  model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

  def _call() -> str:
    """
    Make a single LLM API call.
    
    This is kept as a separate function so that retry_with_backoff()
    can execute the same API call again if a temporary error occurs.
    """

    response = client.chat.completions.create(
      model=model,
      messages=[
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
      ],
      temperature=0.2,
    )

    # Extract and clean the generated answer.
    return (response.choices[0].message.content or "").strip()
  try:
    # Retry temporary API Failures up to 3 times using exponential backoff.
    answer = retry_with_backoff(
      _call,
      max_retries=3,
      base_delay=1.0,
      exceptions=(APIError, APITimeoutError, RateLimitError, TimeoutError),
    )
    logger.info("LLM generation succeeded | model=%s", model)
    return answer
  except Exception as exc:
    # Log the error and pass it to the callers for further handling.
    logger.error("LLM generation failed: %s", exc)
    raise