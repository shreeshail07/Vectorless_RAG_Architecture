
"""
Insurance Claims Investigation Assistant - Streamlit UI

Vectorless retrieval demo: Exact Match + Metadata Filters + BM25 > LLM
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import get_cached_documents
from src.pipeline import run_pipeline
from src.retriever import CLAIM_STATUSES, LOSS_TYPES, US_STATES
from src.utils import logger


st.set_page_config(
    page_title="Insurance Claims Investigation Assistant",
    page_icon="🔍",
    layout="wide",
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


def render_sidebar() -> dict:

    st.sidebar.header("Controls")

    if st.sidebar.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_debug = None
        st.rerun()

    top_k = st.sidebar.slider(
        "Top-K retrieval",
        min_value=1,
        max_value=15,
        value=5,
    )

    st.sidebar.markdown("### Optional filters")

    status = st.sidebar.selectbox(
        "Claim status",
        options=["(any)"] + CLAIM_STATUSES,
    )

    loss_type = st.sidebar.selectbox(
        "Loss type",
        options=["(any)"] + LOSS_TYPES,
    )

    state = st.sidebar.selectbox(
        "State",
        options=["(any)"] + US_STATES,
    )

    siu = st.sidebar.selectbox(
        "SIU referred",
        options=["(any)", "Yes", "No"],
    )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Retrieval is vectorless: exact IDs, metadata filters, and BM25. "
        "No embeddings or vector database."
    )

    return {
        "top_k": top_k,
        "sidebar_filters": {
            "claim_status": None if status == "(any)" else status,
            "loss_type": None if loss_type == "(any)" else loss_type,
            "state": None if state == "(any)" else state,
            "siu_referred": None if siu == "(any)" else siu,
        },
    }


def render_retrieval_debug(response) -> None:

    retrieval = response.retrieval

    st.markdown("### Retrieval")

    cols = st.columns(5)

    cols[0].metric("Exact Match", retrieval.exact_count)
    cols[1].metric("BM25 Results", retrieval.bm25_count)
    cols[2].metric("Filtered Results", retrieval.filtered_count)
    cols[3].metric("Final Context", len(retrieval.documents))
    cols[4].metric("Latency", f"{retrieval.latency_seconds:.2f}s")

    with st.expander("How was this answer retrieved?", expanded=False):

        filters = retrieval.filters

        st.markdown("**Query Understanding**")

        st.json({
            "resolved_query": response.resolved_query,
            "claim_ids": filters.claim_ids,
            "policy_ids": filters.policy_ids,
            "claim_status": filters.claim_status,
            "loss_type": filters.loss_type,
            "state": filters.state,
            "siu_referred": filters.siu_referred,
            "min_paid_amount": filters.min_paid_amount,
            "max_paid_amount": filters.max_paid_amount,
        })

        st.markdown("**Final documents passed to the LLM**")

        for i, item in enumerate(retrieval.documents, start=1):

            meta = item.document.get("metadata") or {}

            st.markdown(
                f"**{i}. {item.document.get('source')}** - "
                f"Methods: `{','.join(item.methods)}` | "
                f"Score: `{item.score:.3f}`"
                + (
                    f" | BM25: `{item.bm25_score:.3f}`"
                    if item.bm25_score is not None
                    else ""
                )
            )

            detail = {
                "claim_id": meta.get("claim_id"),
                "page": meta.get("page"),
                "claim_status": meta.get("claim_status"),
                "loss_type": meta.get("loss_type"),
                "state": meta.get("state"),
            }

            st.write({
                k: v for k, v in detail.items()
                if v not in (None, "")
            })

            st.caption(item.document.get("text", "")[:400] + "...")

    with st.expander("Source / Evidence", expanded=True):

        if not response.citations:
            st.write("No sources retrieved")

        for cite in response.citations:

            line = f"- **{cite['source']}**"

            if cite.get("claim_id"):
                line += f" | Claim ID: `{cite['claim_id']}`"

            if cite.get("page"):
                line += f" | Page: {cite['page']}"

            st.markdown(line)

    if response.llm_latency_seconds or response.total_latency_seconds:

        st.caption(
            f"LLM latency: {response.llm_latency_seconds:.2f}s | "
            f"Total latency: {response.total_latency_seconds:.2f}s"
        )


def main() -> None:

    init_state()

    controls = render_sidebar()

    st.title("Insurance Claims Investigation Assistant")

    st.caption(
        "Production-inspired teaching demo - Vectorless RAG "
        "(Exact Match + Metadata Filters + BM25 + LLM Generation)"
    )

    try:
        n_docs = warm_documents()
        st.sidebar.success(f"Loaded {n_docs} searchable documents")

    except Exception as exc:
        st.error(f"Failed to load data files: {exc}")
        logger.exception("Document loading failed")
        st.stop()

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask about a claim, SIU referral, or policy coverage..."
    )

    if prompt:

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):

            with st.spinner("Retrieving evidence and generating answer..."):

                try:

                    response = run_pipeline(
                        prompt,
                        chat_history=st.session_state.messages[:-1],
                        top_k=controls["top_k"],
                        sidebar_filters=controls["sidebar_filters"],
                    )

                    st.markdown(response.answer)

                    render_retrieval_debug(response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                    })

                    st.session_state.last_debug = response

                    if response.error:
                        logger.warning(
                            "Pipeline returned error flag: %s",
                            response.error,
                        )

                except Exception as exc:

                    logger.exception("Unhandled UI error")

                    st.error(
                        "Something went wrong while processing your question. "
                        f"Details: {exc}"
                    )


if __name__ == "__main__":
    main()
