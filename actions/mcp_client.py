"""
mcp_client.py — generic MCP client for JARVIS

A small wrapper around the official `mcp` Python SDK (Streamable HTTP
transport) that:
  1. Connects to any remote MCP server given a URL + auth headers
  2. Lists that server's tools and converts each one's JSON Schema into
     the format Gemini's function-calling expects (TOOL_DECLARATIONS shape)
  3. Calls a named tool with arguments and returns a plain string result,
     the same shape every other action in JARVIS returns

This is intentionally generic — not GitHub-specific — so any other MCP
server (remote or local) can reuse it later without duplicating this
connection/conversion logic. See github_mcp.py for the GitHub-specific
wiring (token, URL, tool-name namespacing).

Install: pip install mcp httpx
"""

from typing import Any


def _json_schema_to_gemini(schema: dict) -> dict:
    """
    Convert MCP/JSON Schema into Gemini function-declaration format.

    Handles:
      - string / number / integer / boolean / array / object
      - nullable schemas such as ["string", "null"]
      - nested objects
      - arrays
      - descriptions
      - required fields
      - enums
    """

    if not schema:
        return {"type": "OBJECT", "properties": {}}

    type_map = {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    result: dict[str, Any] = {}

    json_type = schema.get("type", "object")

    # JSON Schema allows:
    #   "type": "string"
    # and:
    #   "type": ["string", "null"]
    if isinstance(json_type, list):
        non_null_types = [
            t for t in json_type
            if t != "null"
        ]

        json_type = (
            non_null_types[0]
            if non_null_types
            else "string"
        )

    result["type"] = type_map.get(
        json_type,
        "STRING"
    )

    if "description" in schema:
        result["description"] = schema["description"]

    # Object
    if json_type == "object":

        if "properties" in schema:
            result["properties"] = {
                k: _json_schema_to_gemini(v)
                for k, v in schema["properties"].items()
            }

        if "required" in schema:
            result["required"] = schema["required"]

    # Array
    if json_type == "array" and "items" in schema:
        result["items"] = _json_schema_to_gemini(
            schema["items"]
        )

    # Enum
    if "enum" in schema:
        result["enum"] = schema["enum"]

    return result


async def list_mcp_tools(url: str, headers: dict | None = None) -> list[dict]:
    """
    Connects to the MCP server at *url*, lists its tools, and returns them
    already converted into Gemini TOOL_DECLARATIONS format:
    [{"name": ..., "description": ..., "parameters": {...}}, ...]
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    declarations = []
    async with streamablehttp_client(url, headers=headers or {}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            for tool in result.tools:
                declarations.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _json_schema_to_gemini(tool.inputSchema or {}),
                })
    return declarations


async def call_mcp_tool(url: str, tool_name: str, args: dict,
                         headers: dict | None = None) -> str:
    """
    Connects to the MCP server at *url*, calls *tool_name* with *args*,
    and returns the result as a plain string — the same shape every other
    JARVIS action returns, so it can go straight into a FunctionResponse.

    Opens a fresh connection per call rather than holding one open for the
    app's whole lifetime — simpler, and safe for occasional tool calls
    like GitHub actions (not latency-critical like the voice audio path).
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers=headers or {}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)

    if result.isError:
        text_parts = [c.text for c in result.content if hasattr(c, "text")]
        return f"MCP tool '{tool_name}' returned an error: {' '.join(text_parts) or 'unknown error'}"

    text_parts = [c.text for c in result.content if hasattr(c, "text")]
    return "\n".join(text_parts) if text_parts else "Done (no text content returned)."