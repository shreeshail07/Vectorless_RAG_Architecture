from pathlib import Path

from src.data_loader import load_pdf


def test_pdf_loader_extracts_all_claim_id_formats(tmp_path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "claims_report.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(
        50,
        750,
        "Claims: CLM-1023, CLM-2045 and CLM-4099",
    )
    pdf.save()

    documents = load_pdf(pdf_path)

    assert len(documents) == 1

    claim_ids = documents[0]["metadata"]["claim_ids"]

    assert claim_ids == [
        "CLM-1023",
        "CLM-2045",
        "CLM-4099",
    ]
from src.data_loader import _chunk_claim_notes


def test_claim_notes_ignore_empty_sections():
    raw = (
        "CLM-1023\nInvestigation details for claim 1023."
        "\n--------------------\n"
        "   \n"
        "--------------------\n"
        "CLM-2045\nInvestigation details for claim 2045."
    )

    documents = _chunk_claim_notes(raw, "claim_notes.txt")

    assert len(documents) == 2
    assert documents[0]["metadata"]["claim_id"] == "CLM-1023"
    assert documents[1]["metadata"]["claim_id"] == "CLM-2045"
