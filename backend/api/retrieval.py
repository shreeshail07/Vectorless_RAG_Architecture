import logging

from fastapi import APIRouter, HTTPException

from backend.schemas.claims import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedDocumentResponse,
)
from src.retriever import retrieve

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Claims Retrieval"],
)


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
def retrieve_claim_documents(
    request: RetrievalRequest,
) -> RetrievalResponse:
    """
    Retrieve relevant insurance claim documents using Vectorless RAG.
    """

    try:
        # Convert validated Pydantic filters into a dictionary.
        sidebar_filters = (
            request.filters.model_dump(
                exclude_none=True,
                exclude_defaults=True,
            )
            if request.filters
            else None
        )

        # Call the existing retrieval engine.
        retrieval_result = retrieve(
            query=request.query,
            top_k=request.top_k,
            sidebar_filters=sidebar_filters,
        )

        # Convert internal ScoredDocument objects into API response objects.
        response_documents = []

        for item in retrieval_result.documents:
            document = item.document

            response_documents.append(
                RetrievedDocumentResponse(
                    text=document.get("text", ""),
                    source=document.get("source", ""),
                    doc_type=document.get("doc_type"),
                    metadata=document.get("metadata") or {},
                    score=item.score,
                    methods=item.methods,
                    bm25_score=item.bm25_score,
                )
            )

        return RetrievalResponse(
            query=request.query,
            total_results=len(response_documents),
            exact_count=retrieval_result.exact_count,
            bm25_count=retrieval_result.bm25_count,
            filtered_count=retrieval_result.filtered_count,
            latency_seconds=retrieval_result.latency_seconds,
            results=response_documents,
        )

    except Exception:
        logger.exception("Claim document retrieval failed")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during document retrieval.",
        )