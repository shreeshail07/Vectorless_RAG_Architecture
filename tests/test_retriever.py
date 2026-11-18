import pytest
from src.retriever import retrieve
from src.retriever import (
    BM25Index,
    QueryFilters,
    exact_match,
    metadata_filter,
    parse_query,
)


@pytest.fixture
def sample_documents():
    return [
        {
            "text": "Claim CLM-1023 policy POL-4582. Closed collision claim in Texas. Kitchen fire damage investigation.",
            "source": "claims.csv",
            "doc_type": "claim_record",
            "metadata": {
                "claim_id": "CLM-1023",
                "policy_id": "POL-4582",
                "claim_status": "Closed",
                "loss_type": "Collision",
                "state": "Texas",
                "siu_referred": "No",
                "paid_amount": 50000.0,
            },
        },
        {
            "text": "Claim CLM-2045 policy POL-6789. Open theft claim in Florida.",
            "source": "claims.csv",
            "doc_type": "claim_record",
            "metadata": {
                "claim_id": "CLM-2045",
                "policy_id": "POL-6789",
                "claim_status": "Open",
                "loss_type": "Theft",
                "state": "Florida",
                "siu_referred": "Yes",
                "paid_amount": 12000.0,
            },
        },

                {
            "text": "Claim CLM-3099 policy POL-9876. Pending comprehensive claim in Ohio involving water damage.",
            "source": "claims.csv",
            "doc_type": "claim_record",
            "metadata": {
                "claim_id": "CLM-3099",
                "policy_id": "POL-9876",
                "claim_status": "Pending",
                "loss_type": "Comprehensive",
                "state": "Ohio",
                "siu_referred": "No",
                "paid_amount": 8000.0,
            },
        },
    ]


def test_parse_query_extracts_structured_filters():
    filters = parse_query(
        "Find closed collision claims in Texas above $50,000 CLM-1023 POL-4582"
    )

    assert filters.claim_status == "Closed"
    assert filters.loss_type == "Collision"
    assert filters.state == "Texas"
    assert filters.min_paid_amount == 50000.0
    assert "CLM-1023" in filters.claim_ids
    assert "POL-4582" in filters.policy_ids


def test_exact_match_finds_claim_id(sample_documents):
    filters = QueryFilters(claim_ids=["CLM-1023"])

    results = exact_match(sample_documents, filters)

    assert len(results) == 1
    assert results[0].document["metadata"]["claim_id"] == "CLM-1023"
    assert "exact_claim_id" in results[0].methods


def test_exact_match_finds_policy_id(sample_documents):
    filters = QueryFilters(policy_ids=["POL-6789"])

    results = exact_match(sample_documents, filters)

    assert len(results) == 1
    assert results[0].document["metadata"]["policy_id"] == "POL-6789"
    assert "exact_policy_id" in results[0].methods


def test_metadata_filter_returns_matching_claim(sample_documents):
    filters = QueryFilters(
        claim_status="Closed",
        state="Texas",
    )

    results = metadata_filter(sample_documents, filters)

    assert len(results) == 1
    assert results[0]["metadata"]["claim_id"] == "CLM-1023"


def test_bm25_ranks_relevant_document_first(sample_documents):
    index = BM25Index(sample_documents)

    results = index.search("closed collision Texas", top_k=2)

    assert len(results) >= 1
    assert results[0].document["metadata"]["claim_id"] == "CLM-1023"
    assert "bm25" in results[0].methods

def test_metadata_filter_does_not_return_unrelated_claims(sample_documents):
    filters = QueryFilters(
        claim_status="Closed",
        state="California",
    )

    results = metadata_filter(sample_documents, filters)

    assert results == []

def test_retrieve_does_not_return_unrelated_claims(sample_documents, monkeypatch):
    from src import retriever

    monkeypatch.setattr(
        retriever,
        "get_cached_documents",
        lambda data_dir_str=None: tuple(sample_documents),
    )

    monkeypatch.setattr(
        retriever,
        "get_bm25_index",
        lambda data_dir_str=None: BM25Index(sample_documents),
    )

    result = retrieve(
        query="Find closed claims in California",
        top_k=5,
    )

    assert result.filtered_count == 0
    assert result.documents == []

def test_exact_match_respects_structured_filters(sample_documents, monkeypatch):
    from src import retriever

    monkeypatch.setattr(
        retriever,
        "get_cached_documents",
        lambda data_dir_str=None: tuple(sample_documents),
    )

    monkeypatch.setattr(
        retriever,
        "get_bm25_index",
        lambda data_dir_str=None: BM25Index(sample_documents),
    )

    result = retrieve(
        query="Find closed claims in California CLM-1023",
        top_k=5,
    )

    assert result.documents == []