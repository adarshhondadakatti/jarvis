"""
core/tool_resilience.py

Two independent safety nets for Jarvis's tool-calling layer:

1. run_resilient() — wraps a blocking tool call with a timeout + retry policy.
   Use this for anything that touches the network or an external service
   (Gmail API, weather, flight search, web search, YouTube, etc).

2. ConfirmationGate — server-enforced two-step confirmation for destructive
   actions (deleting/moving files, sending email, shutting the assistant down).
   This does NOT rely on the model honestly setting a "confirmed": true flag
   on its first call. A destructive call is only executed if an *earlier*
   call with the same tool + same key arguments was already seen and the
   user has since had a chance to respond. Concretely:

       call #1 (no confirmed flag, or confirmed flag ignored) -> gate creates
           a pending ticket, tool is NOT executed, Jarvis is told to ask the
           user to confirm out loud.
       call #2 (confirmed=True) -> gate checks a pending ticket exists for the
           same tool + same key args, created within the TTL window -> allow
           execution and clear the ticket.
       call #2 arrives with confirmed=True but NO matching pending ticket
           (model tried to skip the ask) -> still blocked; a fresh ticket is
           created instead, forcing the ask to actually happen.

This means the only way a destructive action executes is if the exact same
request was seen twice, which in practice means the model asked and the user
said yes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("jarvis.tool_resilience")

# ─────────────────────────────────────────────────────────────────────────
# 1. Retry + timeout wrapper
# ─────────────────────────────────────────────────────────────────────────

class ToolError(Exception):
    """Raised when a tool exhausts its retries. Carries the last real error."""
    def __init__(self, tool_name: str, attempts: int, last_error: Exception):
        self.tool_name = tool_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Tool '{tool_name}' failed after {attempts} attempt(s): {last_error}")


class ToolTimeout(ToolError):
    pass

# Errors worth retrying — network/IO flakiness. Anything else (bad args,
# permission errors, missing files) fails fast since retrying won't help.
_TRANSIENT_ERROR_HINTS = (
    "timeout", "timed out", "connection", "temporarily unavailable",
    "rate limit", "429", "503", "502", "reset by peer", "network",
)

def _looks_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _TRANSIENT_ERROR_HINTS)

async def run_resilient(
    fn: Callable[[], Any],
    *,
    name: str,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    timeout: float = 20.0,
    retries: int = 2,
    backoff: float = 1.5,
) -> Any:
    """
    Run a blocking, zero-arg callable `fn` (typically a lambda wrapping an
    action module call) in the default executor, with a timeout per attempt
    and retry-with-backoff on transient failures.

    Non-transient exceptions (bad input, permission errors, etc) are raised
    immediately without retrying.
    """
    loop = loop or asyncio.get_event_loop()
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 2):  # e.g. retries=2 -> up to 3 attempts
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=timeout)
        except asyncio.TimeoutError as e:
            last_error = e
            log.warning(f"[{name}] attempt {attempt} timed out after {timeout}s")
        except Exception as e:
            last_error = e
            if not _looks_transient(e):
                log.warning(f"[{name}] non-transient error, not retrying: {e}")
                raise
            log.warning(f"[{name}] attempt {attempt} failed (transient): {e}")

        if attempt <= retries:
            await asyncio.sleep(backoff * attempt)

    if isinstance(last_error, asyncio.TimeoutError):
        raise ToolTimeout(name, retries + 1, last_error)
    raise ToolError(name, retries + 1, last_error or Exception("unknown error"))


# ─────────────────────────────────────────────────────────────────────────
# 2. Confirmation gate for destructive actions
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _Ticket:
    key: str
    created_at: float = field(default_factory=time.monotonic)

class ConfirmationGate:
    """Server-side pending-action tracker. One instance per JarvisLive session."""

    def __init__(self, ttl_seconds: float = 120.0):
        self._pending: dict[str, _Ticket] = {}
        self._ttl = ttl_seconds

    @staticmethod
    def _key(tool_name: str, args: dict, *, key_fields: tuple[str, ...]) -> str:
        payload = {k: args.get(k) for k in key_fields}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return f"{tool_name}:" + hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _purge_expired(self):
        now = time.monotonic()
        expired = [k for k, t in self._pending.items() if now - t.created_at > self._ttl]
        for k in expired:
            del self._pending[k]

    def check(
        self,
        tool_name: str,
        args: dict,
        *,
        key_fields: tuple[str, ...],
        summary: str,
    ) -> tuple[bool, str]:
        """
        Returns (should_execute, message_for_model).

        should_execute=False means: do NOT run the underlying tool. The
        message tells the model to relay the confirmation question to the
        user verbatim (in the user's own words/language) and wait for a yes.
        """
        self._purge_expired()
        key = self._key(tool_name, args, key_fields=key_fields)
        confirmed_flag = bool(args.get("confirmed"))

        ticket = self._pending.get(key)

        if confirmed_flag and ticket is not None:
            # Real confirmation: same action was asked about and is still fresh.
            del self._pending[key]
            return True, "confirmed"

        # Either no prior ask, or model tried to skip straight to confirmed=True
        # without us ever having asked — in both cases, (re)issue the ask.
        self._pending[key] = _Ticket(key=key)
        msg = (
            f"[CONFIRMATION_REQUIRED] {summary} "
            f"Ask the user to explicitly confirm before proceeding, in their own "
            f"language. Do NOT proceed. Only call this tool again with "
            f"confirmed=true after the user clearly says yes."
        )
        return False, msg