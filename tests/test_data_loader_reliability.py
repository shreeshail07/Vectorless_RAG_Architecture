
from pathlib import Path

import src.data_loader as data_loader


def test_load_all_documents_continues_after_file_failure(
    tmp_path,
    monkeypatch,
):
    # Arrange: create three supported files.
    (tmp_path / "01_claims.csv").touch()
    (tmp_path / "02_broken.pdf").touch()
    (tmp_path / "03_notes.txt").touch()

    def fake_csv_loader(path):
        return [{"text": "Valid CSV claim"}]

    def fake_pdf_loader(path):
        raise ValueError("Simulated corrupted PDF")

    def fake_txt_loader(path):
        return [{"text": "Valid TXT notes"}]

    monkeypatch.setattr(
        data_loader,
        "load_csv_claims",
        fake_csv_loader,
    )
    monkeypatch.setattr(
        data_loader,
        "load_pdf",
        fake_pdf_loader,
    )
    monkeypatch.setattr(
        data_loader,
        "load_text",
        fake_txt_loader,
    )

    # Act
    documents = data_loader.load_all_documents(tmp_path)

    # Assert
    assert len(documents) == 2
    assert documents[0]["text"] == "Valid CSV claim"
    assert documents[1]["text"] == "Valid TXT notes"


def test_load_all_documents_raises_when_no_documents_loaded(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "broken.pdf").touch()

    def fake_pdf_loader(path):
        raise ValueError("Simulated corrupted PDF")

    monkeypatch.setattr(
        data_loader,
        "load_pdf",
        fake_pdf_loader,
    )

    import pytest

    with pytest.raises(ValueError, match="No documents loaded"):
        data_loader.load_all_documents(tmp_path)
