
import logging

import src.data_loader as data_loader


def test_ingestion_logs_success_metrics(tmp_path, monkeypatch, caplog):
    (tmp_path / "claims.csv").touch()
    (tmp_path / "notes.txt").touch()

    def fake_csv_loader(path):
        return [
            {"text": "Claim CLM-1023"},
            {"text": "Claim CLM-2045"},
        ]

    def fake_txt_loader(path):
        return [{"text": "Investigation notes"}]

    monkeypatch.setattr(data_loader, "load_csv_claims", fake_csv_loader)
    monkeypatch.setattr(data_loader, "load_text", fake_txt_loader)

    with caplog.at_level(logging.INFO, logger="claims_assistant"):
        documents = data_loader.load_all_documents(tmp_path)

    assert len(documents) == 3

    summary = next(
        record.getMessage()
        for record in caplog.records
        if "Ingestion summary" in record.getMessage()
    )

    assert "total_files=2" in summary
    assert "successful_files=2" in summary
    assert "failed_files=0" in summary
    assert "total_documents=3" in summary
    assert "duration_seconds=" in summary


def test_ingestion_logs_failed_file_metrics(tmp_path, monkeypatch, caplog):
    (tmp_path / "claims.csv").touch()
    (tmp_path / "broken.pdf").touch()

    def fake_csv_loader(path):
        return [{"text": "Claim CLM-1023"}]

    def fake_pdf_loader(path):
        raise ValueError("Simulated corrupted PDF")

    monkeypatch.setattr(data_loader, "load_csv_claims", fake_csv_loader)
    monkeypatch.setattr(data_loader, "load_pdf", fake_pdf_loader)

    with caplog.at_level(logging.INFO, logger="claims_assistant"):
        documents = data_loader.load_all_documents(tmp_path)

    assert len(documents) == 1

    summary = next(
        record.getMessage()
        for record in caplog.records
        if "Ingestion summary" in record.getMessage()
    )

    assert "total_files=2" in summary
    assert "successful_files=1" in summary
    assert "failed_files=1" in summary
    assert "total_documents=1" in summary

    assert "broken.pdf" in caplog.text
