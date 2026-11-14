"""
Insurance Claims Investigation Assistant- Streamlit UI

Vectorless retrieval demo: Exact Match + Metadata Filters + BM25 > LLM

"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run app.py`

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from src.data_loader import get_cached_documents
from src.pipeline import run_pipeline
from src.retriever import CLAIM_STATUSES, LOSS_TYPES, US_STATES
from src.utils import logger

st.set_page_config(
  page_title="Insurance Claims Investigation Assistant",
  page_icon="*",
  layout = "wide",
)

@st.cache_resource
def warm_documents() -> int:
  docs = get_cached_documents()
  return len(docs)

def init_state() -> None:
  if "messages" not in st.session_state:
    st.session_state.messages = []
  if "last_debug" not in st.session_state:
    st.session_state.last_debug = None
