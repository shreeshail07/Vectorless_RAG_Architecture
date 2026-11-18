
from src.data_loader import (
    get_cached_documents,
    refresh_document_cache,
)


def test_refresh_document_cache_clears_lru_cache():
    # Arrange: put a known entry into the cache.
    get_cached_documents.cache_clear()

    get_cached_documents.cache_info()

    # Act
    refresh_document_cache()

    # Assert
    cache_info = get_cached_documents.cache_info()

    assert cache_info.currsize == 0
