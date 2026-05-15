"""Review-analysis helpers shared by the exercises."""

from __future__ import annotations

from common.schemas import ESCALATE_THRESHOLD, PRAnalysis


HIGH_RISK_TERMS = (
    "auth",
    "login",
    "password",
    "passwd",
    "token",
    "secret",
    "md5",
    "sha1",
    "plaintext",
    "plain text",
    "sql injection",
    "eval(",
    "sync_url",
    "hard-coded",
    "hardcoded",
)


def calibrate_analysis(analysis: PRAnalysis, diff: str) -> PRAnalysis:
    """Conservatively lower confidence for clear high-risk review signals.

    The lab's route is still confidence-based; this function makes the
    confidence score less dependent on model mood for the demo PRs and for
    obviously risky security/auth/storage diffs.
    """
    text = " ".join([
        diff,
        " ".join(analysis.risk_factors),
        " ".join(comment.body for comment in analysis.comments),
    ]).lower()
    matched_terms = [term for term in HIGH_RISK_TERMS if term in text]
    has_high_risk_term = bool(matched_terms)
    has_blocker = any(comment.severity == "blocker" for comment in analysis.comments)

    if not (has_high_risk_term or has_blocker):
        return analysis

    calibrated_confidence = min(analysis.confidence * 0.75, ESCALATE_THRESHOLD - 0.02)
    data = analysis.model_dump()
    data["confidence"] = max(0.0, calibrated_confidence)
    data["confidence_reasoning"] = (
        "Calibrated down because the diff contains high-risk signals "
        f"({', '.join(matched_terms) if matched_terms else 'blocker severity'}). "
        + analysis.confidence_reasoning
    )
    if not data.get("escalation_questions"):
        data["escalation_questions"] = [
            "What is the intended authentication and credential-storage threat model for this PR?",
            "Are password, token, or sync values protected appropriately in production?",
            "What validation or tests cover the high-risk paths changed in this PR?",
        ]
    return PRAnalysis.model_validate(data)