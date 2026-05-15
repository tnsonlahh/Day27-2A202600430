"""Exercise 5 - Streamlit approval UI for the HITL PR review agent."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


def init_state() -> None:
    defaults = {
        "thread_id": None,
        "pr_url": "",
        "interrupt_payload": None,
        "final": None,
        "error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


async def list_recent_sessions() -> list[dict[str, Any]]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   CASE MAX(
                       CASE risk_level
                           WHEN 'high' THEN 3
                           WHEN 'med' THEN 2
                           ELSE 1
                       END
                   )
                       WHEN 3 THEN 'high'
                       WHEN 2 THEN 'med'
                       ELSE 'low'
                   END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Recent sessions")
        try:
            sessions = asyncio.run(list_recent_sessions())
        except Exception as exc:
            st.caption(f"Audit database unavailable: {exc}")
            return

        if not sessions:
            st.caption("No sessions yet.")
            return

        labels = [
            f"{row['last_event']} | {row['worst_risk']} | {row['thread_id'][:8]}"
            for row in sessions
        ]
        selected = st.selectbox("Session", labels, index=0)
        row = sessions[labels.index(selected)]
        st.caption(row["pr_url"])
        st.caption(f"events={row['events']} thread={row['thread_id']}")
        if st.button("Load session"):
            st.session_state.thread_id = row["thread_id"]
            st.session_state.pr_url = row["pr_url"]
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.session_state.error = None
            st.rerun()


def render_approval_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Approval requested - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    comments = payload.get("comments", [])
    if comments:
        st.markdown("#### Proposed comments")
        for comment in comments:
            st.markdown(
                f"- **[{comment['severity']}]** "
                f"`{comment['file']}:{comment.get('line') or '?'}` - {comment['body']}"
            )

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_area("Feedback", key="approval_feedback", height=96)
    col1, col2, col3 = st.columns(3)
    reviewer_id = os.environ.get("GITHUB_USER")
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback, "reviewer_id": reviewer_id}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback, "reviewer_id": reviewer_id}
    if col3.button("Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback, "reviewer_id": reviewer_id}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Strong escalation - confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for index, question in enumerate(payload.get("questions", []), start=1):
            answers[question] = st.text_area(f"Q{index}: {question}", key=f"esc_{index}", height=88)
        submitted = st.form_submit_button("Submit answers", type="primary")
    return answers if submitted else None


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        app = build_graph(checkpointer)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


def handle_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def start_review(pr_url: str) -> None:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None

    with st.spinner("Fetching PR and calling the LLM..."):
        try:
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
            handle_result(result)
        except Exception as exc:
            st.session_state.error = str(exc)


def resume_review(answer: dict) -> None:
    st.session_state.error = None
    with st.spinner("Resuming graph..."):
        try:
            result = asyncio.run(run_graph(
                st.session_state.pr_url,
                st.session_state.thread_id,
                resume_value=answer,
            ))
            handle_result(result)
        except Exception as exc:
            st.session_state.error = str(exc)


st.set_page_config(page_title="HITL PR Review", layout="wide")
init_state()

st.title("HITL PR Review Agent")
render_sidebar()

with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")

if submitted and pr_url:
    start_review(pr_url)
    st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)

payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    if kind == "approval_request":
        answer = render_approval_card(payload)
    elif kind == "escalation":
        answer = render_escalation_card(payload)
    else:
        st.error(f"Unknown interrupt kind: {kind}")
        answer = None

    if answer is not None:
        resume_review(answer)
        st.rerun()

if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")

    if action in {"auto_approved", "committed", "committed_with_edits", "committed_after_escalation"}:
        st.success(f"{action} - comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected - no comment posted")
    elif action == "commit_failed":
        st.error("Commit failed - check GITHUB_TOKEN and PR permissions")
    else:
        st.info(f"final_action = {action}")

    if analysis is not None:
        st.markdown(analysis.summary)
        if final.get("initial_confidence") is not None and final.get("escalation_answers"):
            st.caption(
                f"initial confidence = {final['initial_confidence']:.0%} | "
                f"final confidence = {analysis.confidence:.0%}"
            )
        else:
            st.caption(f"final confidence = {analysis.confidence:.0%}")

    st.caption(
        f"thread_id = {st.session_state.thread_id} | "
        f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )