"""
Data loading for CSV, TXT, PDF AND DOCX into a common document format.

each document is a dict:
{
"text": "...",
"source": "claim_notes.text",
"metadata": {...}
}
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from pypdf import PdfReader

from src.utils import logger

DocumentDict = dict[str, Any]

def _project_root() -> Path:
  return Path(__file__).resolve().parent.parent

def default_data_dir() -> Path:
  return _project_root() / "data"


def load_csv_claims(path: Path) -> list[DocumentDict]:
  """
  Load claims from a csv file and convert each row into a searchable document.
  
  The CSV values are normal so  they can be reliable used for
  filtering and retrieval later.
  
  Example: 
      claim.csv
      Each row > searchable document with text + metadata
      
      list[DocumentDict]
  """
  df = pd.read_csv(path)
  documents: list[DocumentDict] =[]

  for _, row in df.iterrows():
    # Start with all csv columns as metadata
    metadata = {col: row[col] for col in df.columns}

    # Normalize important fields for consistent filtering and matching.
    metadata["claim_id"] =str(metadata.get("claim_id", "")).upper()
    metadata["policy_id"] = str(metadata.get("policy_id", "")).upper()
    metadata["claim_status"] = str(metadata.get("claim_status", ""))
    metadata["loss_type"] = str(metadata.get("loss_type",""))
    metadata["state"] = str(metadata.get("state",""))
    metadata["siu_referred"] = str(metadata.get("siu_referred"))


    #make sure paid_amount is always stored as a number.
    try:
      metadata["paid_amount"] = float(metadata.get("paid_amount", 0) or 0)
    except (TypeError, ValueError):
      metadata["paid_amount"] = 0.0

    # Create a natural-language representation of the structured claim.
    # this text can letter be searched using BM25

    text =(
      f"Claim {metadata['claim_id']} on policy {metadata['policy_id']}."
      f"Status{metadata['claim_status']}. Loss type: {metadata['loss_type']}."
      f"Loss date: {metadata.get('loss_date', '')}.State: {metadata['state']}."
      f"Paid amount: ${metadata['paid_amount']:,.2f}."
      f"Claimant: {metadata.get('claimant_name','')}."
      f"Vehicle: {metadata.get('vehicle_type','')}."
      f"SIU referred: {metadata['siu_referred']}."
      f"SIU referral reason: {metadata.get('siu_referral_reason','') or 'N/A'}."

    )

    # Store the searchable text along with its metadata and source.
    documents.append(
      {
        "text": text,
        "source": path.name,
        "doc_type": "claim_record",
        "metadata": metadata,
      }
    )
  return documents

def load_text(path: Path) -> list[DocumentDict]:
  """
  Load a TXT file and split it into smaller searchable documents.
  Claim notes use a custom chunking function, while other text files
  are split using blank lines. Claim IDs are also extracted as metadata.
  
  Example:
      claim_notes.txt > claim-specific chunks
      siu_guidelines.txt > paragraph-based chunks
  """

  raw =path.read_text(encoding="utf-8")

  # Claim notes have a specific structure, so custom chunking
  if path.name =="claim_notes.txt":
    return _chunk_claim_notes(raw, path.name)

  # Split general text files into paragraphs using blank lines.

  chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]

  documents: list[DocumentDict] = []

  for i, chunk in enumerate(chunks, start=1):
    # Extract a Claim ID if the chunk contains one.
    claim_ids = re.findall(r"CLM-\d{4}", chunk, flags=re.IGNORECASE)

    documents.append(
      {
        "text": chunk,
        "source": path.name,
        "doc_type": "text",
        "metadata": {
          "chunk_id": i,
          "claim_id": claim_ids[0].upper() if claim_ids else"",
        },
      }
    )
  return documents

def _chunk_claim_notes(raw: str, source: str) -> list[DocumentDict]:
  """
  Spilt claim notes into separate searchable documents.
  
  Each section is expected to represent one claim. The function
  extracts the Claim ID and stores it as metadata.
  
  Example :
      Raw notes containing:
      CLM-1001
      Investigation details...
      > One document with claim_id = "CLM-1001"
  
  """

  # claim notes are separated by a line containing 10+ hyphens.
  parts = [p.strip() for p in re.split(r"\n-{10,}\n", raw) if p.strip]

  documents: list[DocumentDict] =[]

  for i, part in enumerate(parts, start=1):
    # Remove any synthetic/header text before the actual claim content.
    if part.startswith("SYNTHETIC") or part.startswith("="):
      match_inline = re.search(
        r"(Claim\s+CLM-\d{4}|CLM-d{4})",
        part,
        flags=re.IGNORECASE,
      )

      if not match_inline:
        continue
      part = part[match_inline.start():].strip()

    # Extract the claim ID from the note.
    match = re.search(r"CLM-\d{4}", part, flags=re.IGNORECASE)
    claim_id =  match.group(0).upper() if  match else ""

    #ignore sections that do not belong to a specific claim.
    if not claim_id:
      continue

    documents.append(
      {
        "text": part,
        "source": source,
        "doc_type": "claim_note",
        "metadata": {
          "chunk_id": i,
          "claim_id": claim_id,
        },
      }
    )

  return documents

def load_pdf(path: Path) -> list[DocumentDict]:
  """
  Load a PDF and convert each page into a searchable document.
  
  Each page keeps its page number and any Claim IDs found in te text
  as metadata, making it possible to cite the correct source later.
  
  Example:
      investigation_report.pdf
      
      page 1 > document.
      page 2 > document.
      page 3 > document.
  """

  reader = PdfReader(str(path))
  documents: list[DocumentDict]=[]

  for page_num, page in enumerate(reader.pages, start=1):
    # Extract text from the current PDF page.
    text = (page.extract_text() or "").strip()

    # Skip pages that contain no extractable text.
    if not text:
      continue

    # Find all Claim IDs mentioned on this page.
    claim_ids = re.findall(r"CLM-4\d{4}", text, flags=re.IGNORECASE)

    documents.append(
      {
        "text":text,
        "source": path.name,
        "doc_type": "pdf_page",
        "metadata": {
          "page": page_num,
          "claim_ids": claim_ids[0].upper() if claim_ids else "",
          "claim_ids": [c.upper() for c in claim_ids],
         },
      }
    )
  return documents
