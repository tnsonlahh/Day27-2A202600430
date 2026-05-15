"""Exercise 4 - Structured SQLite audit trail + durable checkpointer."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_conn, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.review import calibrate_analysis
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


def elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def make_entry(
    *,
    action: str,
    confidence: float,
    decision: str,
    reason: str | None,
    execution_time_ms: int,
    reviewer_id: str | None = None,
    risk_level: str | None = None,
) -> AuditEntry:
    return AuditEntry(
        agent_id=AGENT_ID,
        action=action,
        confidence=confidence,
        risk_level=risk_level or risk_level_for(confidence),
        reviewer_id=reviewer_id,
        decision=decision,
        reason=reason,
        execution_time_ms=execution_time_ms,
    )


async def audit(state: ReviewState, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the audit_events table."""
    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
    )


async def audit_interrupt_pending_once(state: ReviewState, entry: AuditEntry) -> None:
    """Write the pre-interrupt pending row once per thread/action.

    LangGraph re-runs an interrupted node from the top after Command(resume=...).
    Without this guard, the required pre-interrupt audit row would be duplicated.
    """
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT 1
              FROM audit_events
             WHERE thread_id = ?
               AND pr_url = ?
               AND action = ?
               AND decision = 'pending'
             LIMIT 1
            """,
            (state["thread_id"], state["pr_url"], entry.action),
        ) as cur:
            if await cur.fetchone():
                return
    await audit(state, entry)


async def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]ok[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(state, make_entry(
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=elapsed_ms(t0),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis: PRAnalysis = await llm.ainvoke([
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer reviewing a pull request. "
                    "Return structured PRAnalysis only. Include actionable comments "
                    "with file paths and line numbers when possible. If confidence "
                    "is below 60%, include 2-4 specific escalation_questions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n"
                    f"Files: {', '.join(state.get('pr_files', []))}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    analysis = calibrate_analysis(analysis, state["pr_diff"])
    console.print(
        f"  [green]ok[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.comments)} comment(s)"
    )
    await audit(state, make_entry(
        action="analyze",
        confidence=analysis.confidence,
        decision="pending",
        reason=analysis.confidence_reasoning,
        execution_time_ms=elapsed_ms(t0),
    ))
    return {
        "analysis": analysis,
        "initial_confidence": analysis.confidence,
        "initial_confidence_reasoning": analysis.confidence_reasoning,
    }


async def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    t0 = time.monotonic()
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(
        f"  [green]ok[/green] decision=[bold]{decision}[/bold] "
        f"(confidence={confidence:.0%})"
    )
    await audit(state, make_entry(
        action="route",
        confidence=confidence,
        decision=decision,
        reason=f"Routed by confidence {confidence:.0%}",
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state: ReviewState) -> dict:
    console.print("[cyan]-> human_approval[/cyan]")
    t0 = time.monotonic()
    analysis = state["analysis"]

    await audit_interrupt_pending_once(state, make_entry(
        action="human_approval",
        confidence=analysis.confidence,
        decision="pending",
        reason=analysis.confidence_reasoning,
        execution_time_ms=elapsed_ms(t0),
    ))

    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": analysis.confidence,
        "confidence_reasoning": analysis.confidence_reasoning,
        "summary": analysis.summary,
        "comments": [comment.model_dump() for comment in analysis.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    choice = str(response.get("choice", "")).lower()
    if choice not in {"approve", "reject", "edit"}:
        choice = "reject"
    feedback = response.get("feedback") or None
    reviewer_id = response.get("reviewer_id") or os.environ.get("GITHUB_USER")

    await audit(state, make_entry(
        action="human_approval",
        confidence=analysis.confidence,
        decision=choice,
        reason=feedback or f"Reviewer chose {choice}",
        reviewer_id=reviewer_id,
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"human_choice": choice, "human_feedback": feedback}


def _render_comment_body(state: ReviewState) -> str:
    analysis = state["analysis"]
    initial_confidence = state.get("initial_confidence")
    if state.get("escalation_answers") and initial_confidence is not None:
        title = (
            "### HITL refined review "
            f"(initial {initial_confidence:.0%}, final {analysis.confidence:.0%})"
        )
    else:
        title = f"### Automated review (confidence {analysis.confidence:.0%})"
    lines = [title, "", analysis.summary, ""]
    for comment in analysis.comments:
        lines.append(
            f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` - {comment.body}"
        )
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for question, answer in state["escalation_answers"].items():
            lines.append(f"> **{question}** {answer}")
    return "\n".join(lines)


def _post(state: ReviewState) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]ok[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as exc:
        console.print(f"  [red]failed[/red] post failed: {exc}")
        return "commit_failed"


async def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]-> commit[/cyan]")
    t0 = time.monotonic()
    analysis = state["analysis"]

    if state.get("escalation_answers"):
        post_result = _post(state)
        final_action = "committed_after_escalation" if post_result == "committed" else post_result
        decision = "escalate"
    elif state.get("human_choice") in {"approve", "edit"}:
        post_result = _post(state)
        if post_result == "committed" and state.get("human_choice") == "edit":
            final_action = "committed_with_edits"
        else:
            final_action = post_result
        decision = state.get("human_choice") or "approve"
    else:
        console.print(f"  [yellow]skip[/yellow] no comment posted (choice={state.get('human_choice')})")
        final_action = "rejected"
        decision = "reject"

    await audit(state, make_entry(
        action="commit",
        confidence=analysis.confidence,
        decision=decision,
        reason=f"final_action={final_action}",
        reviewer_id=os.environ.get("GITHUB_USER") if decision in {"approve", "edit", "reject"} else None,
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"final_action": final_action, "posted_comment_body": _render_comment_body(state)}


async def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan] [dim]high confidence - posting directly[/dim]")
    t0 = time.monotonic()
    analysis = state["analysis"]
    post_result = _post(state)
    final_action = "auto_approved" if post_result == "committed" else post_result
    await audit(state, make_entry(
        action="auto_approve",
        confidence=analysis.confidence,
        decision="auto",
        reason=f"final_action={final_action}",
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"final_action": final_action, "posted_comment_body": _render_comment_body(state)}


async def node_escalate(state: ReviewState) -> dict:
    console.print("[cyan]-> escalate[/cyan]")
    t0 = time.monotonic()
    analysis = state["analysis"]
    questions = analysis.escalation_questions or [
        "What is the intended risk profile of this PR?",
        "Are there migration, security, or rollout constraints not visible in the diff?",
    ]

    await audit_interrupt_pending_once(state, make_entry(
        action="escalate",
        confidence=analysis.confidence,
        decision="pending",
        reason=analysis.confidence_reasoning,
        execution_time_ms=elapsed_ms(t0),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": analysis.confidence,
        "confidence_reasoning": analysis.confidence_reasoning,
        "summary": analysis.summary,
        "risk_factors": analysis.risk_factors,
        "questions": questions,
        "diff_preview": state["pr_diff"][:2000],
    })

    if not isinstance(answers, dict):
        answers = {}
    reviewer_id = os.environ.get("GITHUB_USER")
    await audit(state, make_entry(
        action="escalate",
        confidence=analysis.confidence,
        decision="escalate",
        reason=f"Reviewer answered {len(answers)} escalation question(s)",
        reviewer_id=reviewer_id,
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state: ReviewState) -> dict:
    console.print("[cyan]-> synthesize[/cyan]")
    t0 = time.monotonic()
    initial = state["analysis"]
    qa = "\n".join(
        f"Q: {question}\nA: {answer}"
        for question, answer in (state.get("escalation_answers") or {}).items()
    )
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {
                "role": "system",
                "content": (
                    "Refine a pull-request review using reviewer answers. Return "
                    "structured PRAnalysis only. Preserve true issues, remove resolved "
                    "uncertainties, and produce actionable final comments."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PR title: {state['pr_title']}\n"
                    f"Initial confidence: {initial.confidence:.0%}\n"
                    f"Initial summary: {initial.summary}\n"
                    f"Initial reasoning: {initial.confidence_reasoning}\n"
                    f"Initial risk factors: {initial.risk_factors}\n\n"
                    f"Reviewer Q&A:\n{qa or '(no answers provided)'}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]ok[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, make_entry(
        action="synthesize",
        confidence=refined.confidence,
        decision="escalate",
        reason=refined.confidence_reasoning,
        reviewer_id=os.environ.get("GITHUB_USER"),
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload: dict):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approval request confidence={payload['confidence']:.0%}",
            border_style="green",
        ))
        for comment in payload.get("comments", []):
            console.print(
                f"  [{comment['severity']}] {comment['file']}:"
                f"{comment.get('line') or '?'} - {comment['body']}"
            )
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        return {
            "choice": choice,
            "feedback": feedback,
            "reviewer_id": os.environ.get("GITHUB_USER"),
        }

    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation confidence={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        if payload.get("risk_factors"):
            console.print("[red]Risk factors:[/red] " + ", ".join(payload["risk_factors"]))
        return {
            question: console.input(f"Q: {question}\nA: ").strip()
            for question in payload["questions"]
        }

    raise ValueError(f"Unknown interrupt kind: {kind}")


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 - SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        app = build_graph(checkpointer)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    parser.add_argument("--thread", help="Use a specific LangGraph thread id")
    args = parser.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()