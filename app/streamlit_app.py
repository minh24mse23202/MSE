from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("ARAGBIZ_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Adaptive RAG Workflow QA", layout="wide")
st.title("Adaptive RAG Workflow QA")

question = st.text_input("Business workflow question", value="How should we handle an invoice mismatch after goods are received?")

if st.button("Ask", type="primary") and question.strip():
    response = requests.post(f"{API_URL}/answer", json={"question": question}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    st.subheader("Answer")
    st.write(payload["answer"])

    metadata = payload["metadata"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Complexity", metadata["complexity_label"])
    metric_cols[1].metric("Route", metadata["retrieval_mode"])
    metric_cols[2].metric("Top K", metadata["top_k"])
    metric_cols[3].metric("Latency ms", metadata["latency_ms"])

    st.subheader("Retrieved Contexts")
    for context in payload["contexts"]:
        with st.expander(f"#{context['rank']} {context['id']} score={context['score']:.3f}"):
            st.write(context["text"])
            st.json(context["metadata"])

    st.subheader("Feedback")
    rating = st.radio("Was this answer useful?", ["up", "down"], horizontal=True)
    comment = st.text_area("Comment", placeholder="Optional feedback")
    if st.button("Submit feedback"):
        requests.post(
            f"{API_URL}/feedback",
            json={
                "question": payload["question"],
                "answer": payload["answer"],
                "rating": rating,
                "comment": comment,
                "metadata": payload["metadata"],
            },
            timeout=10,
        ).raise_for_status()
        st.success("Feedback recorded")

