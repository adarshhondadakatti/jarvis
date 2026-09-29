"""
github_mcp.py — GitHub integration for JARVIS via GitHub's official
remote MCP server (https://api.githubcopilot.com/mcp/).

Unlike every other actions/*.py file, this one doesn't hardcode a fixed
set of operations. GitHub maintains the tool list on their server — this
module discovers whatever tools are currently available (list PRs, create
issues, check workflow runs, search code, etc.) at JARVIS startup and
hands them to main.py to register with Gemini dynamically.

Setup:
  1. Create a GitHub Personal Access Token: https://github.com/settings/personal-access-tokens/new
  2. Add it to config/api_keys.json as "github_token": "ghp_..."
  3. pip install mcp httpx
  4. See main.py integration notes at the bottom of this file.

Where to see exactly what tools you have access to: this module logs
every discovered tool's name and one-line description to
jarvis_debug.log on startup (same debug log the file summarizer uses) —
open that file after starting JARVIS to see your actual available tool
list, since it depends on your token's scopes. GitHub's own reference
docs are also at https://github.com/github/github-mcp-server (README +
docs/ folder), organized by "toolset" (repos, issues, pull_requests,
actions, code_security, etc.).
"""

import json
import time
from pathlib import Path

from actions.mcp_client import list_mcp_tools, call_mcp_tool

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

# Tools discovered from GitHub get this prefix in Gemini's tool list, so
# JARVIS can tell "this is a GitHub MCP tool" apart from its own hardcoded
# tools at dispatch time without needing a separate registry lookup.
_TOOL_PREFIX = "github_"

_DEBUG_LOG = Path(__file__).resolve().parent.parent / "jarvis_debug.log"


def _unwrap_exception(e: BaseException, depth: int = 0) -> list[str]:
    """
    anyio/mcp wrap real errors inside ExceptionGroup/TaskGroup, so a plain
    str(e) just says "unhandled errors in a TaskGroup" and hides the actual
    cause. Recursively unwraps nested groups and returns a flat list of
    "ClassName: message" strings for every real exception found inside.
    """
    lines = []
    prefix = "  " * depth
    if hasattr(e, "exceptions"):  # ExceptionGroup / BaseExceptionGroup
        lines.append(f"{prefix}{type(e).__name__}: {e} (unwrapping {len(e.exceptions)} sub-error(s))")
        for sub in e.exceptions:
            lines.extend(_unwrap_exception(sub, depth + 1))
    else:
        lines.append(f"{prefix}{type(e).__name__}: {e}")
    return lines


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [github_mcp] {msg}"
    print(line)
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def _get_github_token() -> str | None:
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("github_token")
    except Exception:
        return None


def _headers() -> dict:
    token = _get_github_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


async def discover_github_tools() -> list[dict]:
    """
    Called once at JARVIS startup. Fetches GitHub's current tool list,
    converts each to Gemini's declaration format, and prefixes each name
    with "github_" so main.py can route calls back here. Returns [] (and
    logs why) if no token is configured or the server can't be reached —
    JARVIS should still start normally in that case, just without GitHub
    tools available.
    """
    token = _get_github_token()
    if not token:
        _log("No github_token in config/api_keys.json — GitHub tools not loaded.")
        return []

    try:
        raw_tools = await list_mcp_tools(GITHUB_MCP_URL, headers=_headers())
    except Exception as e:
        _log("Could not reach GitHub MCP server. Full error breakdown:")
        for line in _unwrap_exception(e):
            _log(f"  {line}")
        return []

    declarations = []
    for tool in raw_tools:
        prefixed = {**tool, "name": f"{_TOOL_PREFIX}{tool['name']}"}
        declarations.append(prefixed)

    _log(f"Discovered {len(declarations)} GitHub tools:")
    for tool in declarations:
        _log(f"  - {tool['name']}: {tool['description'][:100]}")

    return declarations


def is_github_tool(name: str) -> bool:
    """Used by main.py's dispatcher to check if a Gemini function call is a GitHub one."""
    return name.startswith(_TOOL_PREFIX)


async def call_github_tool(name: str, args: dict) -> str:
    """
    Called by main.py when Gemini invokes a tool whose name starts with
    'github_'. Strips the prefix back off and forwards to the real MCP
    tool name on GitHub's server.
    """
    real_name = name[len(_TOOL_PREFIX):]
    _log(f"Calling '{real_name}' with {args}")
    try:
        result = await call_mcp_tool(GITHUB_MCP_URL, real_name, args, headers=_headers())
        _log(f"'{real_name}' -> {result[:200]}")
        return result
    except Exception as e:
        _log(f"'{real_name}' failed: {e}")
        return f"GitHub action '{real_name}' failed: {e}"


# ── main.py integration ──────────────────────────────────────────────────
#
# 1. In JarvisLive.run(), right after `self._loop = asyncio.get_event_loop()`
#    and before the reconnect while-loop, fetch the tools ONCE per process:
#
#        from actions.github_mcp import discover_github_tools
#        self._github_tool_declarations = await discover_github_tools()
#
#    (self._github_tool_declarations should default to [] in __init__ too,
#    in case run() hasn't reached this line yet when _build_config() is
#    first called.)
#
# 2. In _build_config(), extend the existing tools list so Gemini sees
#    both the hardcoded TOOL_DECLARATIONS and whatever GitHub exposed:
#
#        tools=[{"function_declarations": TOOL_DECLARATIONS + self._github_tool_declarations}]
#
#    (replaces the current `tools=[{"function_declarations": TOOL_DECLARATIONS}]` line)
#
# 3. In _execute_tool(), add a branch before the final `else` that catches
#    everything else, so GitHub calls route here instead of falling to
#    "Unknown tool":
#
#        elif name.startswith("github_"):
#            from actions.github_mcp import call_github_tool
#            result = await call_github_tool(name, args)
#
#    (async already — call_github_tool can be awaited directly, no
#    run_in_executor needed, since it's already async I/O)