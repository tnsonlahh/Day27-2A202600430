"""Exercise 3 - Escalation branch with reviewer Q&A."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.review import calibrate_analysis
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]ok[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer reviewing a pull request. "
                    "Return structured PRAnalysis only. If confidence is below 60%, "
                    "populate escalation_questions with 2-4 specific, context-rich "
                    "questions that reference affected files or diff sections. "
                    "Security, auth, persistence, migrations, external network calls, "
                    "and missing tests should reduce confidence."
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
        f"{len(analysis.escalation_questions)} question(s)"
    )
    return {
        "analysis": analysis,
        "initial_confidence": analysis.confidence,
        "initial_confidence_reasoning": analysis.confidence_reasoning,
    }


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
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
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions and store the answers."""
    console.print("[cyan]-> escalate[/cyan]")
    analysis = state["analysis"]
    questions = analysis.escalation_questions or [
        "What is the intended risk profile of this PR?",
        "Are there migration, security, or rollout constraints not visible in the diff?",
    ]
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
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt the real LLM with reviewer answers to produce a refined review."""
    console.print("[cyan]-> synthesize[/cyan]")
    answers = state.get("escalation_answers") or {}
    qa = "\n".join(f"Q: {question}\nA: {answer}" for question, answer in answers.items())
    initial = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are refining a pull-request review after a human answered "
                    "escalation questions. Return structured PRAnalysis only. Use "
                    "the answers as authoritative context, preserve valid initial "
                    "findings, remove concerns that the answers resolve, and add "
                    "specific actionable comments for remaining risks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PR title: {state['pr_title']}\n"
                    f"Initial summary: {initial.summary}\n"
                    f"Initial confidence: {initial.confidence:.0%}\n"
                    f"Initial reasoning: {initial.confidence_reasoning}\n"
                    f"Initial risk factors: {initial.risk_factors}\n\n"
                    f"Reviewer Q&A:\n{qa or '(no answers provided)'}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]ok[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[cyan]-> human_approval[/cyan]")
    analysis = state["analysis"]
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
    return {
        "human_choice": choice,
        "human_feedback": response.get("feedback") or None,
    }


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


def _post(state: ReviewState, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]ok[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as exc:
        console.print(f"  [red]failed[/red] post failed: {exc}")
        return "commit_failed"


def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]-> commit[/cyan]")
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") in {"approve", "edit"}:
        label = "committed" if state.get("human_choice") == "approve" else "committed_with_edits"
        return {"final_action": _post(state, label)}
    console.print(f"  [yellow]skip[/yellow] no comment posted (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan] [dim]high confidence - posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
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
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload: dict):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? confidence={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        return {"choice": choice, "feedback": feedback}

    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation confidence={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        if payload.get("risk_factors"):
            console.print("[red]Risk factors:[/red] " + ", ".join(payload["risk_factors"]))
        return {question: console.input(f"Q: {question}\nA: ").strip() for question in payload["questions"]}

    raise ValueError(f"Unknown interrupt kind: {kind}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 3 - escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = app.invoke(Command(resume=handle_interrupt(payload)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        if result.get("initial_confidence") is not None and result.get("escalation_answers"):
            console.print(
                f"initial confidence = {result['initial_confidence']:.0%} | "
                f"final confidence = {result['analysis'].confidence:.0%}"
            )
        else:
            console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()