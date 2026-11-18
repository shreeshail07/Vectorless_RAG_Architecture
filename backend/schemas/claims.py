
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ClaimFiltersRequest(BaseModel):
    """Optional structured filters for insurance claim retrieval."""

    claim_ids: list[str] = Field(default_factory=list)
    policy_ids: list[str] = Field(default_factory=list)

    claim_status: str | None = None
    loss_type: str | None = None
    state: str | None = None
    siu_referred: str | None = None

    min_paid_amount: float | None = Field(default=None, ge=0)
    max_paid_amount: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_paid_amount_range(self):
        if (
            self.min_paid_amount is not None
            and self.max_paid_amount is not None
            and self.min_paid_amount > self.max_paid_amount
        ):
            raise ValueError(
                "min_paid_amount cannot be greater than max_paid_amount"
            )
        return self


class RetrievalRequest(BaseModel):
    """Request body for the document retrieval endpoint."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: ClaimFiltersRequest | None = None

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be empty or contain only spaces")
        return value


class RetrievedDocumentResponse(BaseModel):
    """One document returned by the retrieval engine."""

    text: str
    source: str
    doc_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    score: float
    methods: list[str]
    bm25_score: float | None = None


class RetrievalResponse(BaseModel):
    """Structured response from the retrieval pipeline."""

    query: str
    total_results: int
    exact_count: int
    bm25_count: int
    filtered_count: int
    latency_seconds: float
    results: list[RetrievedDocumentResponse]

