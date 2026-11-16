
from src.retriever import (
    get_bm25_index,
    refresh_retrieval_caches,
)


def test_refresh_rebuilds_bm25_index_with_updated_documents(tmp_path):
    report = tmp_path / "investigation.txt"

    try:
        # 1. Create the original investigation report.
        report.write_text(
            "Investigation report for CLM-1023. "
            "Water damage assessment completed.",
            encoding="utf-8",
        )

        # 2. Build the original BM25 index.
        old_index = get_bm25_index(str(tmp_path))

        assert any(
            "CLM-1023" in doc["text"]
            for doc in old_index.documents
        )

        # 3. Update the investigation report.
        report.write_text(
            "Investigation report for CLM-2045. "
            "Fire damage assessment completed.",
            encoding="utf-8",
        )

        # 4. Clear both document and BM25 caches.
        refresh_retrieval_caches()

        # 5. Build a new BM25 index.
        new_index = get_bm25_index(str(tmp_path))

        # 6. Verify the refreshed index.
        assert new_index is not old_index

        assert any(
            "CLM-2045" in doc["text"]
            for doc in new_index.documents
        )

        assert not any(
            "CLM-1023" in doc["text"]
            for doc in new_index.documents
        )

    finally:
        refresh_retrieval_caches()
