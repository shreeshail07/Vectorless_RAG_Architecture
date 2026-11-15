"""
Vectorless retrieval: exact match + metadata filtering = BM25 + result fusion.

No embeddings. NO vector database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from src.data_loader import DocumentDict, get_cached_documents
from src.utils import (
  extract_claim_ids,
  extract_policy_ids,
  logger,
  tokenize,
)

CLAIM_STATUSES = ["Open", "Closed", "Under Review", "Denied", "Pending"]
LOSS_TYPES = ["Collision", "Comprehensive", "Theft", "Fire", "Flood", "Vandalism"]
US_STATES = [
  "Texas", "California", "Florida", "New York", "Illinois", "Ohio", "Georgia",
  "Arizona", "North Carolina", "Michigan", "Pennsylvania", "Colorado",
  "Washington", "Nevada", "Tennessee",
]

@dataclass
class QueryFilters:
  claim_ids: list[str] = field(default_factory=list)
  policy_ids: list[str] = field(default_factory=list)
  claim_status: str | None = None
  loss_type: str | None = None
  state: str | None = None
  siu_referred: str | None = None
  min_paid_amount: float | None = None
  max_paid_amount: float | None = None
  free_text: str = ""

@dataclass
class ScoredDocument:
  document: DocumentDict
  score: float
  methods: list[str]
  bm25_score: float | None = None
@dataclass
class RetrievalResult:
  documents: list[ScoredDocument]
  filters: QueryFilters
  exact_count:int
  bm25_count:int
  filtered_count: int
  latency_seconds: float

def parse_query(
    query: str, sidebar_filters: dict[str, Any] | None = None
) -> QueryFilters:
  """
  Convert a natural-language question into structured search filters.
  
  It uses simple rules to detect Claim IDs, policy IDs, status, loss type,
  state, SIU conditions and payments amounts.
  
  Example:
    "Find closed collision claims in Texas above $50, 5000"
    
    claim_status="Closed"
    loss_type= "Collision"
    state= "Texas"
    mini_paid_amount = 50000
  """

  sidebar_filters = sidebar_filters or {}
  q = query.strip()
  lower = q.lower()

  filters = QueryFilters(
    claim_ids=extract_claim_ids(q),
    policy_ids=extract_policy_ids(q),
    free_text=q,
  )

  # Look for a known claim status such as "closed" or "Open"

  for status in CLAIM_STATUSES:
    if status.lower() in lower:
      filters.claim_status = status
      break

  # Detect loss type unless the question is clearly about policy wording.
  policy_context = any(
    phrase in lower 
    for phrase in (
      "coverage",
      "deductible",
      "exclusion",
      "policy document",
      "what does",
    )
  )

  if not policy_context:
    for loss in LOSS_TYPES:
      if loss.lower() in lower:
        filters.loss_type = loss
        break
  # Look for a US state mentioned in the question.
  for state in US_STATES:
    if re.search(rf"\b{re.escape(state.lower())}\b", lower):
      filters.state = state
      break
  # Detect whether the query implies an SIU referral.
  if "siu" in lower or "special investigation" in lower:
    if any(w in lower for w in ("referred", "referral", "fraud", "because")):
      filters.siu_referred = "Yes"
    elif "not referred" in lower or "no siu" in lower:
      filters.siu_referred = "No"

  # Extract a minimum paid amount from phrases like "above $50,000"
  amount_match = re.search(
    r"(?:above|over|greater than|more than)\s*\$?\s*([\d,]+(?:\.\d+)?)",
    lower,
  )

  if amount_match:
    filters.min_paid_amount = float(amount_match.group(1).replace(",",""))

  #extract a maximum paid amount form phrases like "below $10,000"
  amount_below = re.search(
    r"(?:below|less than)\s*\$?\s*([\d,]+(?:\.\d+)?)",
    lower,
  )

  if amount_below:
    filters.max_paid_amount=float(amount_below.group(1).replace(",",""))

  # Explicit sidebar filters take priority over filters from the query
  if sidebar_filters.get("claim_status"):
    filters.claim_status=sidebar_filters["claim_status"]
  if sidebar_filters.get("loss_type"):
    filters.loss_type=sidebar_filters["loss_type"]
  if sidebar_filters.get("state"):
    filters.state=sidebar_filters["state"]
  if sidebar_filters.get("siu_referred"):
    filters.siu_referred=sidebar_filters["siu_referred"]
  return filters

def _meta(doc: DocumentDict) -> dict[str, Any]:
  return doc.get("metadata") or {}

def exact_match(
    documents: list[DocumentDict], filters: QueryFilters
) -> list[ScoredDocument]:
  """
  Find documents containing exact claim ids or policy ids from the query.
  
  Exact matching is useful when the user provides a specific identifier,
  because it is more reliable than keyword-based search.
  
  Example:
      Query: "Show me CLM-1023"
      > Documents containing CLM-1023 are returned with high score
  """
  results: list[ScoredDocument] =[]

  # Normalize ids so matching is case-insensitive.
  target_claims = {c.upper() for c in filters.claim_ids}
  target_policies = {p.upper() for p in filters.policy_ids}

  # No ids means there is nothing for exact matching to search for.
  if not target_claims and not target_policies:
    return results
  for doc in documents:
    meta = _meta(doc)
    text_upper = doc["text"].upper()
    matched= False
    methods: list[str] =[]

    claim_id = str(meta.get("claim_id", "")).upper()
    policy_id = str(meta.get("policy_id", "")).upper()

    # some documents may contain multiple claim ids.
    extra_claims = {
      str(c).upper() 
      for c in meta.get("claim_ids", []) or []
    }

    # Check both metadata and the actual document text for claim ids.
    if target_claims and(
      claim_id in target_claims
      or bool(target_claims & extra_claims)
      or any(cid in text_upper for cid in target_claims)
    ):
      matched = True
      methods.append("exact_claim_id")

    # Check both metadata and document text for Policy IDs.
    if target_policies and (
      policy_id in target_policies
      or any(pid in text_upper for pid in target_policies)
    ):
      matched = True
      methods.append("exact_policy_id")

    if matched:
      results.append(
        ScoredDocument(
          document=doc,
          score=100.0,
          methods=methods,
        )
      )
  return results
def metadata_filter(
    documents: list[DocumentDict], filters: QueryFilters
)-> list[DocumentDict]:
  """
  Filters documents using structured claim metadata.
  
  if first identifies which claim records satisfy the requested filters,
  then keeps those claims and their related notes/reports as supporting context.
  
  
  Example:
      Query: "closed claims in Texas above $50,5000"
      > keeps matching claim records and their related documents.
  """


  # Check whether the query contains any structured filters.
  has_structured = any(
    [
      filters.claim_status,
      filters.loss_type,
      filters.state,
      filters.siu_referred,
      filters.min_paid_amount is not None,
      filters.max_paid_amount is not None,
      filters.claim_ids,
      filters.policy_ids,
    ]
  )
  # No Structured filters means every document remains eligible.

  if not has_structured:
    return list(documents)

  # Claim records contain the structured fields used for filtering.
  claim_rows = [d for d in documents if d.get("doc_type") == "claim_record"]
  matched_claim_ids: set[str] = set()

  for doc in claim_rows:
    meta = _meta(doc)

    # Apply each requested filter. If any filter fails, skip this claim.

    if filters.claim_ids and meta.get("claim_id", "").upper() not in {
      c.upper() for c in filters.claim_ids
    }:
      continue
    if filters.policy_ids and meta.get("policy_id", "").upper() not in {
      p.upper() for p in filters.policy_ids
    }:
      continue

    if filters.claim_status and meta.get("claim_status","") != filters.claim_status:
      continue

    if filters.loss_type and meta.get("loss_type","") != filters.loss_type:
      continue
    if filters.state and meta.get("state", "") != filters.state:
      continue
    if filters.siu_referred and meta.get("siu_referred") != filters.siu_referred:
      continue
    paid = float(meta.get("paid_amount") or 0)

    if filters.min_paid_amount is not None and paid < filters.min_paid_amount:
      continue
    if filters.max_paid_amount is not None and paid > filters.max_paid_amount:
      continue
    # This claim passed every requested filter.
    matched_claim_ids.add(str(meta.get("claim_id", "").upper()))

  # If there are no field based filters, leave the documents unchanged.
  if not matched_claim_ids and not(
    filters.claim_status
    or filters.loss_type
    or filters.state
    or filters.siu_referred
    or filters.min_paid_amount is not None
    or filters.max_paid_amount is not None
  ):
    return list(documents)

  filtered: list[DocumentDict]=[]

  for doc in documents:
    meta = _meta(doc)
    claim_id = str(meta.get("claim_id","")).upper()

    # Some documents, such as PDF pages, can reference multiple claims.
    extra ={ str(c).upper() 
            for c in meta.get("claim_ids", []) or []}

    if doc.get("doc_type") == "claim_record":
      # keep only claims records that passed the filters.
      if claim_id in matched_claim_ids:
        filtered.append(doc)
        continue

      # also keep notes / reports linked to matching claims.
      # General guidelines and policy documents can provide supporting context
      # even when they have no Claim ID.
      if (
        matched_claim_ids
        and (
          claim_id in matched_claim_ids
          or (extra & matched_claim_ids)
          or (doc.get("doc_type") in ("text", "docx_section") and not claim_id)
        )
      ):
        filtered.append(doc)
    else:
      # No claim matched the structured filters.
      continue
  return filtered


class BM25Index:
  """
  Build a BM25 search index that can be reused for multiple queries.
  
  BM25 ranks documents based on keyword frequency and how important those keywords are across the document collection.
  
  Example: 
      Query : "staged accident"
      > Returns documents containing relevant terms,
      ranked by their BM25 score.
  """
  def __init__(self, documents: list[DocumentDict]) -> None:
    self.documents = documents

    # Tokenize every document once when building the index
    self.corpus_tokens = [tokenize(doc["text"]) for doc in documents]

    # Build the BM25 index for fast keyword-based retrieval
    self.bm25= BM25Okapi(self.corpus_tokens)

  def search(self, query:str, top_k:int = 10) -> list[ScoredDocument]:
    """
    Search the index and return the top matching documents.
    
    The query is tokenized, scored against every document
    and ranked from highest to lowest BM25 score.
    """

    tokens = tokenize(query)

    # no searchable terms means no results.
    if not tokens:
      return[]
    # Calculate a BM25 relevance score for every documents
    scores = self.bm25.get_scores(tokens)

    # Rank documents by score, highest first
    ranked = sorted(
      enumerate(scores), key=lambda pair:pair[1], reverse=True
    )

    results: list[ScoredDocument]=[]

    for idx, score in ranked[:top_k]:
      # Ignore documents with no keyword relevance.
      if score <=0:
        continue
      results.append(
        ScoredDocument(
          document=self.documents[idx],
          score=float(score),
          methods=["bm25"],
          bm25_score=float(score),
        )
      )
    return results

def has_structured_filters(filters: QueryFilters) -> bool:
  """
  Check whether  the query contains any structured filters.
  
  Returns True if at least one field such as claim status, state,
  Claim id, or payment amount has been specified.
  
  
  Example:
      state= "Texas"> True
      no filters > False
  """

  return any(
    [
      filters.claim_status,
      filters.loss_type,
      filters.state,
      filters.siu_referred,
      filters.min_paid_amount is not None,
      filters.max_paid_amount is not None,
      filters.claim_ids,
      filters.policy_ids,
    ]
  )
def fuse_results(
    exact: list[ScoredDocument],
    bm25: list[ScoredDocument],
    filtered_docs: list[DocumentDict],
    top_k: int,
    *,
    structured_filters_applied: bool,
) -> list[ScoredDocument]:
  """
  Combine exact-match , BM25 and metadata-filter results into one ranked list.
  
  Exact ID matches receive the highest priority, followed by BM25 results
  that satisfy structured filters. Duplicate documents are merged into one
  result with their retrieval methods combined.
  
  Example:
  Exact match + Bm25 match for the same claim
  one result with both retrieval methods recorded.
  """

  # Track which documents passed metadata filtering.
  filtered_ids = {id(doc) for doc in filtered_docs}

  #store one score document per unique document.
  merged:dict[int, ScoredDocument] ={}

  def upsert(item: ScoredDocument, bonus: float=0.0) -> None:
    """
    Add a result or merge it with an existing result.
    
    if the same document was found by multiple retrieval methods recorded.
    keep one result and recorded all methods that found it.

    """
    key = id(item.document)
    if key not in merged:
      merged[key] = ScoredDocument(
        document=item.document,
        score= item.score + bonus,
        methods=list(item.methods)
        
      )
    else:
      existing = merged[key]

      #keep the stronger score and apply the new bonus.
      existing.score= max(existing.score, item.score) + bonus

      # Record every retrieval method that found this document.
      for method in item.methods:
        if method not in existing.methods:
          existing.methods.append(method)

      if item.bm25_score is not None:
        existing.bm25_score=item.bm25_score
  # exact id matches receive the strongest priority
  for item in exact:
    bonus = (
      20.0
      if (
        not structured_filters_applied or id(item.document) in filtered_ids
      )
      else 0.0

  
    )
    upsert(item, bonus=bonus)

  #BM25 results receive an additional boost when they pass metadata filters
  for item in bm25:
    bonus =(
      5.0
      if structured_filters_applied and id(item.document) in filtered_ids
      else 0.0
    )
    upsert(item, bonus=bonus)

  # Add matching claim records that were found only through metadata filtering.
  if structured_filters_applied:
    for doc in filtered_docs:
      if doc.get("doc_type") != "claim_record":
        continue
      key = id(doc)
      if key not in merged:
        merged[key] = ScoredDocument(
          document=doc,
          score=10.0,
          methods=["metadata_filter"],
        )
      elif "metadata_filter" not in merged[key].methods:
        merged[key].methods.append("metadata_filter")
        merged[key].score += 3.0

  #return the highest -scoring document first.
  ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)

  return ranked[:top_k]
@lru_cache(maxsize=1)
def get_bm25_index(data_dir_str: str |None =None) -> BM25Index:
  """
  Build and cache the BM25 index for the loaded documents.
  
  The index is created once and reused for subsequent searches,
  avoiding the cost of rebuilding it for every query.
  """

  # get the already cached documents instead of loading them from dist again

  docs =list(get_cached_documents(data_dir_str))

  logger.info("Building BM25 index over %s documents", len(docs))

  #build the searchable BM25 index.
  return BM25Index(docs)

def retrieve(
    query: str,
    top_k: int=5,
    sidebar_filters:dict[str, Any] | None = None,
    data_dir_str: str | None = None,
) -> RetrievalResult:
  
  """
  Run the complete vectorless retrieval pipeline.
  
  The query is parsed then exact-matching, metadata filtering, and BM25
  search are performed. Their results are fused into one ranked list.
  
  example;
      Query: "Find closed claims in texas about staged accidents"
      >Parse filters > Metadata filter > BM25 > result fusion.
      > Top 5 relevant documents
  """

  import time
  # Start a timer to measure retrieval performance.
  start = time.perf_counter()

  # Get the cached documents instead of loading files again.
  documents = list(get_cached_documents(data_dir_str))

  # convert the natural-language query into structured filters.
  filters = parse_query(query, sidebar_filters)

  # Run the different vectorless retrieval strategies.
  exact = exact_match(documents, filters)
  structured = has_structured_filters(filters)
  filtered = metadata_filter(documents, filters)

  # When structured filters are available search only the filtered corpus.
  # Otherwise search the complete BM25 index.
  use_filtered_corpus = structured and bool(filtered)

  if use_filtered_corpus:
    bm25_hits = BM25Index(filtered).search(
      query, top_k=max(top_k * 3, 15)
    )
    filtered_count = len(filtered)
  else:
    bm25_hits = get_bm25_index(data_dir_str).search(
      query, top_k=max(top_k * 3, 15)

    )
    # Only report filtered documents when structured filtered was requested.
    filtered_count = len(filtered) if structured else 0
    filtered = filtered if structured else []

  # Combines all retrieval results into one ranked list.
  final = fuse_results(
    exact,
    bm25_hits,
    filtered,
    top_k=top_k,
    structured_filters_applied=structured,
  )
  latency = time.perf_counter()-start

  # Log retrieval statistics for debugging and performance monitoring.
  logger.info(
    "Retrieval | exact=%s bm25=%s filtered=%s final=%s latency=%.3fs query=%r",
    len(exact),
    len(bm25_hits),
    filtered_count,
    len(final),
    latency,
    query,
  )


  return RetrievalResult(
    documents=final,
    filters=filters,
    exact_count=len(exact),
    bm25_count=len(bm25_hits),
    filtered_count=filtered_count,
    latency_seconds=latency,

  )





