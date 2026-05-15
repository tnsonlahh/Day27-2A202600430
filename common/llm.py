"""LLM factory for real OpenAI-compatible chat providers.

The lab scaffold originally targets OpenRouter, but this project can use a
real OpenAI key directly when OPENAI_API_KEY is present in .env. No mock model
or canned response path is used here.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def _openai_model_name() -> str:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


def _openrouter_model_name() -> str:
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    if "/" not in model:
        return f"openai/{model}"
    return model


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Return a real ChatOpenAI client.

    Priority:
    1. OPENAI_API_KEY: call the OpenAI API directly.
    2. OPENROUTER_API_KEY: call OpenRouter through its OpenAI-compatible API.

    Set LLM_MODEL in .env to override the default model.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=_openai_model_name(),
            api_key=openai_key,
            temperature=temperature,
        )

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return ChatOpenAI(
            model=_openrouter_model_name(),
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=openrouter_key,
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set OPENAI_API_KEY for direct OpenAI usage, "
        "or OPENROUTER_API_KEY for OpenRouter."
    )