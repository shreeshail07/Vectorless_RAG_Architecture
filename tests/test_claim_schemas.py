import pytest
from pydantic import ValidationError

from backend.schemas.claims import RetrievalRequest


def test_valid_retrieval_request():
    request = RetrievalRequest(
        query="Find closed water damage claims",
        top_k=5,
        filters={
            "claim_status": "Closed",
            "loss_type": "Water",
        },
    )

    assert request.query == "Find closed water damage claims"
    assert request.top_k == 5
    assert request.filters.claim_status == "Closed"


def test_top_k_uses_default_value():
    request = RetrievalRequest(query="Find water damage claims")

    assert request.top_k == 5


def test_blank_query_is_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="   ")


def test_top_k_above_limit_is_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="Find claims", top_k=51)


def test_invalid_paid_amount_range_is_rejected():
    with pytest.raises(ValidationError):
        RetrievalRequest(
            query="Find claims",
            filters={
                "min_paid_amount": 50000,
                "max_paid_amount": 10000,
            },
        )


def test_valid_paid_amount_range_is_accepted():
    request = RetrievalRequest(
        query="Find claims",
        filters={
            "min_paid_amount": 10000,
            "max_paid_amount": 50000,
        },
    )

    assert request.filters.min_paid_amount == 10000
    assert request.filters.max_paid_amount == 50000
