"""Retry policy for subtask eval runs (transport flakes, empty LLM choices, etc.)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def is_transient_eval_error(message: str | None) -> bool:
    """True for provider/network issues and empty-response bugs that may succeed on retry."""
    if not message:
        return False
    m = message.lower()
    needles = (
        "apierror",
        "internalservererror",
        "openrouterexception",
        "rate limit",
        "rate_limit",
        "429",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "disconnected",
        "reset by peer",
        "chunked read",
        "incomplete chunked",
        "connection reset",
        "try again",
        "server error",
        "service unavailable",
        "api connection",
        "apiconnectionerror",
        "serviceunavailableerror",
        "indexerror",
        "list index out of range",
        "empty choices",
        "litellm",
    )
    return any(n in m for n in needles)


def should_retry_eval_row(row: dict[str, Any], *, max_turns: int) -> bool:
    """Whether to run the same eval again (same logs path, overwrites last attempt)."""
    err = row.get("error")
    sm = row.get("scoring_method", "")
    if is_transient_eval_error(err if err else None):
        return True
    # Early submit_answer with unparseable JSON (truncation / partial tool payload).
    if sm in ("parse_failure", "invalid_submission") and not err:
        ep = row.get("episodes")
        if ep is not None and ep < max_turns:
            return True
    return False


def run_eval_with_retries(
    run_once: Callable[[], dict],
    *,
    eval_retries: int,
    max_turns: int,
) -> dict:
    """Run ``run_once`` until success (no retry) or attempts exhausted.

    ``eval_retries`` is the number of *extra* attempts after the first (default 3 → up to 4 tries).
    """
    if eval_retries < 0:
        eval_retries = 0
    max_attempts = max(1, 1 + eval_retries)
    for attempt in range(1, max_attempts + 1):
        try:
            row = run_once()
        except Exception as exc:
            if attempt >= max_attempts or not is_transient_eval_error(str(exc)):
                raise
            time.sleep(min(2 ** (attempt - 1), 30))
            continue

        if not should_retry_eval_row(row, max_turns=max_turns) or attempt >= max_attempts:
            out = dict(row)
            out["eval_attempt"] = attempt
            return out
        time.sleep(min(2 ** (attempt - 1), 30))
