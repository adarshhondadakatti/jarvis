"""
Microsoft Teams Calendar Module for JARVIS Assistant

API-driven calendar module for Microsoft 365 / Outlook calendar via Microsoft Graph,
with MSAL OAuth 2.0 authentication (delegated, work/school Entra ID account).
Supports listing, creating, updating, and deleting events, plus free/busy lookups.
Events created with is_teams_meeting=True get an auto-generated Microsoft Teams join link.

Setup:
1. Go to the Azure Portal (https://portal.azure.com/) > Entra ID > App registrations
   > New registration.
   - Supported account types: "Accounts in this organizational directory only"
     (or "any organizational directory" if you want multi-tenant).
   - Redirect URI: leave blank here — set it under Authentication (next step).
2. Under Authentication:
   - Add platform > Mobile and desktop applications > check http://localhost
   - Enable "Allow public client flows" > Yes
3. Under API permissions > Add a permission > Microsoft Graph > Delegated permissions:
   - Calendars.ReadWrite
   - User.Read
   - OnlineMeetings.ReadWrite
   Click "Grant admin consent" (a tenant admin must do this for org-wide use).
4. Copy the "Application (client) ID" and "Directory (tenant) ID" from the Overview page.
5. Add them to config/api_keys.json:
   {
     "ms_client_id": "<application-client-id>",
     "ms_tenant_id": "<directory-tenant-id>"
   }
6. First run: say "JARVIS, Teams calendar status" or use the calendar_teams tool with
   action='auth'. A browser window opens for Microsoft sign-in/consent. Tokens are cached
   in config/msgraph_token_cache.json.
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import msal
import requests

from memory.config_manager import CONFIG_DIR, get_ms_client_id, get_ms_tenant_id

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

GRAPH_SCOPES = ["Calendars.ReadWrite", "User.Read", "OnlineMeetings.ReadWrite"]
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

TOKEN_CACHE_FILE = CONFIG_DIR / "msgraph_token_cache.json"

MAX_RETRIES = 3
RETRY_BACKOFF = 2
RATE_LIMIT_DELAY = 1.0

DEFAULT_MAX_RESULTS = 10
# ───────────────────────────────────────────────────────────────────────────────
# Data Classes
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class CalendarEvent:
    """Clean event data structure."""
    id: str
    summary: str
    description: str
    location: str
    start: str            # "YYYY-MM-DDTHH:MM:SS TimeZone"
    end: str
    attendees: list[str] = field(default_factory=list)
    organizer: str = ""
    html_link: str = ""
    meet_link: str = ""
    status: str = ""
    is_all_day: bool = False
# ───────────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────────

class CalendarTeamsError(Exception):
    """Base exception for Teams calendar module errors."""
    pass
class AuthenticationError(CalendarTeamsError):
    """Authentication/authorization failure."""
    pass
class TokenExpiredError(AuthenticationError):
    """Token expired and silent/interactive refresh failed."""
    pass
class EventNotFoundError(CalendarTeamsError):
    """Requested event does not exist."""
    pass
class ConfigurationError(CalendarTeamsError):
    """Azure app registration (client_id/tenant_id) not configured."""
    pass
# ───────────────────────────────────────────────────────────────────────────────
# Token Cache
# ───────────────────────────────────────────────────────────────────────────────

def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_FILE.exists():
        try:
            cache.deserialize(TOKEN_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[CalendarTeams] ⚠️ Failed to load token cache: {e}")
    return cache
def _save_cache(cache: msal.SerializableTokenCache) -> None:
    if not cache.has_state_changed:
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE_FILE.write_text(cache.serialize(), encoding="utf-8")
    try:
        TOKEN_CACHE_FILE.chmod(0o600)
    except Exception:
        pass  # Windows ignores chmod
def _delete_cache() -> None:
    if TOKEN_CACHE_FILE.exists():
        TOKEN_CACHE_FILE.unlink()
# ───────────────────────────────────────────────────────────────────────────────
# Authentication
# ───────────────────────────────────────────────────────────────────────────────

def _get_app() -> tuple[msal.PublicClientApplication, msal.SerializableTokenCache]:
    client_id = get_ms_client_id()
    if not client_id:
        raise ConfigurationError(
            "Microsoft Graph app not configured. Register an app in the Azure Portal "
            "and add ms_client_id (and ms_tenant_id) to config/api_keys.json. "
            "See the module docstring for setup instructions."
        )
    tenant_id = get_ms_tenant_id()
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    cache = _load_cache()
    app = msal.PublicClientApplication(client_id, authority=authority, token_cache=cache)
    return app, cache
def get_graph_token() -> str:
    """
    Get a valid Microsoft Graph access token.
    Tries silent acquisition (cached/refresh token) first, then falls back to an
    interactive browser sign-in.
    """
    app, cache = _get_app()

    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(GRAPH_SCOPES, account=accounts[0])

    if not result:
        result = _run_interactive_auth(app)

    _save_cache(cache)

    if not result or "access_token" not in result:
        err = (result or {}).get("error_description", "Unknown error")
        raise AuthenticationError(f"Failed to acquire Graph token: {err}")

    return result["access_token"]
def _run_interactive_auth(app: msal.PublicClientApplication) -> Optional[dict]:
    """Run interactive (browser) sign-in flow for desktop app."""
    try:
        print("[CalendarTeams] 🌐 Opening browser for Microsoft sign-in...")
        return app.acquire_token_interactive(scopes=GRAPH_SCOPES, port=8082)
    except Exception as e:
        print(f"[CalendarTeams] ❌ Interactive auth failed: {e}")
        traceback.print_exc()
        return None
def force_reauth() -> None:
    """Force re-authentication by clearing the token cache."""
    _delete_cache()
    print("[CalendarTeams] 🔐 Token cache cleared — next operation will require re-authentication")
def is_authenticated() -> bool:
    """Check if a cached account with a silently-refreshable token exists."""
    try:
        app, _ = _get_app()
    except ConfigurationError:
        return False
    accounts = app.get_accounts()
    if not accounts:
        return False
    result = app.acquire_token_silent(GRAPH_SCOPES, account=accounts[0])
    return bool(result and "access_token" in result)
# ───────────────────────────────────────────────────────────────────────────────
# Graph HTTP Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _graph_request(method: str, path: str, retries: int = MAX_RETRIES, **kwargs) -> Any:
    """Execute a Microsoft Graph API request with auth + retry/backoff."""
    url = path if path.startswith("http") else f"{GRAPH_BASE}{path}"

    for attempt in range(retries):
        token = get_graph_token()
        headers = kwargs.pop("headers", {}) or {}
        headers["Authorization"] = f"Bearer {token}"
        headers.setdefault("Content-Type", "application/json")

        resp = requests.request(method, url, headers=headers, timeout=30, **kwargs)

        if resp.status_code in (200, 201, 202):
            return resp.json() if resp.content else {}
        if resp.status_code == 204:
            return {}
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)))
            print(f"[CalendarTeams] ⏳ Rate limited (429), waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        if resp.status_code in (500, 502, 503, 504):
            wait = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
            print(f"[CalendarTeams] ⏳ Server error {resp.status_code}, waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        if resp.status_code == 401:
            raise TokenExpiredError("Graph authentication token expired or invalid")
        if resp.status_code == 403:
            raise AuthenticationError(f"Insufficient Graph permissions: {resp.text[:300]}")
        if resp.status_code == 404:
            raise EventNotFoundError("Event not found")

        raise CalendarTeamsError(f"Graph API error ({resp.status_code}): {resp.text[:300]}")

    raise CalendarTeamsError(f"Max retries ({retries}) exceeded")
# ───────────────────────────────────────────────────────────────────────────────
# Datetime Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _to_graph_datetime(iso_str: str) -> dict:
    """Convert an ISO 8601 datetime (with or without UTC offset) to a Graph
    dateTimeTimeZone resource, normalized to UTC."""
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    else:
        dt = dt.replace(tzinfo=timezone.utc)
    return {"dateTime": dt.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": "UTC"}
def _fmt_graph_datetime(info: dict) -> str:
    if not info:
        return ""
    return f"{info.get('dateTime', '')} {info.get('timeZone', '')}".strip()
# ───────────────────────────────────────────────────────────────────────────────
# Parsing Utilities
# ───────────────────────────────────────────────────────────────────────────────

def _parse_event(item: dict) -> CalendarEvent:
    """Parse a Microsoft Graph event resource into a CalendarEvent."""
    online_meeting = item.get("onlineMeeting") or {}
    body = item.get("body", {}) or {}

    return CalendarEvent(
        id=item.get("id", ""),
        summary=item.get("subject") or "(No title)",
        description=body.get("content", ""),
        location=(item.get("location") or {}).get("displayName", ""),
        start=_fmt_graph_datetime(item.get("start", {})),
        end=_fmt_graph_datetime(item.get("end", {})),
        attendees=[
            a.get("emailAddress", {}).get("address", "")
            for a in item.get("attendees", [])
        ],
        organizer=(item.get("organizer") or {}).get("emailAddress", {}).get("address", ""),
        html_link=item.get("webLink", ""),
        meet_link=online_meeting.get("joinUrl", ""),
        status=item.get("showAs", ""),
        is_all_day=bool(item.get("isAllDay", False)),
    )
# ───────────────────────────────────────────────────────────────────────────────
# Core Calendar Operations
# ───────────────────────────────────────────────────────────────────────────────

def list_events(
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    query: Optional[str] = None,
) -> list[CalendarEvent]:
    """
    List events in the given window via the calendarView endpoint (expands recurring events).

    Args:
        time_min: ISO 8601 datetime with UTC offset (default: now)
        time_max: ISO 8601 datetime with UTC offset (default: 7 days from time_min)
        max_results: Maximum events to return
        query: Free-text filter applied client-side against subject/location

    Returns:
        List of CalendarEvent objects sorted by start time
    """
    if time_min is None:
        time_min = datetime.now(timezone.utc).isoformat()
    if time_max is None:
        time_max = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

    start = _to_graph_datetime(time_min)["dateTime"]
    end = _to_graph_datetime(time_max)["dateTime"]

    print(f"[CalendarTeams] 🔍 Listing events: {start} → {end}, max={max_results}")

    params = {
        "startDateTime": start,
        "endDateTime": end,
        "$top": max_results,
        "$orderby": "start/dateTime",
    }
    headers = {"Prefer": 'outlook.timezone="UTC"'}
    data = _graph_request("GET", "/me/calendarView", params=params, headers=headers)
    items = data.get("value", [])
    events = [_parse_event(item) for item in items]

    if query:
        q = query.lower()
        events = [e for e in events if q in e.summary.lower() or q in e.location.lower()]

    print(f"[CalendarTeams] ✅ Found {len(events)} event(s)")
    return events
def create_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    attendees: Optional[list[str]] = None,
    is_teams_meeting: bool = False,
) -> CalendarEvent:
    """
    Create a calendar event.

    Args:
        summary: Event title
        start_time: ISO 8601 datetime with UTC offset, e.g. "2026-09-22T15:00:00+05:30"
        end_time: ISO 8601 datetime with UTC offset
        description: Event body/notes
        location: Physical location or meeting room
        attendees: List of attendee email addresses
        is_teams_meeting: If True, attaches an auto-generated Microsoft Teams join link

    Returns:
        The created CalendarEvent
    """
    body: dict = {
        "subject": summary,
        "body": {"contentType": "Text", "content": description},
        "location": {"displayName": location},
        "start": _to_graph_datetime(start_time),
        "end": _to_graph_datetime(end_time),
    }
    if attendees:
        body["attendees"] = [
            {"emailAddress": {"address": a}, "type": "required"} for a in attendees
        ]
    if is_teams_meeting:
        body["isOnlineMeeting"] = True
        body["onlineMeetingProvider"] = "teamsForBusiness"

    item = _graph_request("POST", "/me/events", json=body)
    event = _parse_event(item)
    print(f"[CalendarTeams] ✅ Created event '{event.summary}' ({event.id})")
    return event
def update_event(
    event_id: str,
    summary: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[list[str]] = None,
) -> CalendarEvent:
    """Update an existing event. Only non-None fields are changed."""
    body: dict = {}
    if summary is not None:
        body["subject"] = summary
    if description is not None:
        body["body"] = {"contentType": "Text", "content": description}
    if location is not None:
        body["location"] = {"displayName": location}
    if start_time is not None:
        body["start"] = _to_graph_datetime(start_time)
    if end_time is not None:
        body["end"] = _to_graph_datetime(end_time)
    if attendees is not None:
        body["attendees"] = [
            {"emailAddress": {"address": a}, "type": "required"} for a in attendees
        ]

    item = _graph_request("PATCH", f"/me/events/{event_id}", json=body)
    event = _parse_event(item)
    print(f"[CalendarTeams] ✅ Updated event '{event.summary}' ({event.id})")
    return event
def delete_event(event_id: str) -> bool:
    """Delete an event by ID."""
    try:
        _graph_request("DELETE", f"/me/events/{event_id}")
        print(f"[CalendarTeams] ✅ Deleted event {event_id}")
        return True
    except EventNotFoundError:
        print(f"[CalendarTeams] ⚠️ Event {event_id} not found (already deleted?)")
        return False
def get_free_busy(
    time_min: str,
    time_max: str,
    attendees: Optional[list[str]] = None,
) -> dict:
    """
    Return busy time ranges for the given schedules (self + optional attendees)
    between time_min and time_max, via the getSchedule action.

    Returns:
        Dict mapping email -> list of {"start": iso, "end": iso, "status": str} busy intervals
    """
    me = _graph_request("GET", "/me")
    self_email = me.get("mail") or me.get("userPrincipalName", "")
    schedules = list(dict.fromkeys([self_email, *(attendees or [])]))

    body = {
        "schedules": schedules,
        "startTime": _to_graph_datetime(time_min),
        "endTime": _to_graph_datetime(time_max),
        "availabilityViewInterval": 30,
    }
    data = _graph_request("POST", "/me/calendar/getSchedule", json=body)

    result: dict[str, list[dict]] = {}
    for sched in data.get("value", []):
        email = sched.get("scheduleId", "")
        busy = [
            {
                "start": item.get("start", {}).get("dateTime", ""),
                "end": item.get("end", {}).get("dateTime", ""),
                "status": item.get("status", ""),
            }
            for item in sched.get("scheduleItems", [])
        ]
        result[email] = busy
    return result
# ───────────────────────────────────────────────────────────────────────────────
# Tool Interface for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

def calendar_teams_action(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """
    JARVIS tool entry point for Microsoft Teams / Outlook Calendar operations.

    Supported actions:
    - list: List upcoming events (time_min/time_max/max_results/query)
    - create: Create an event (summary, start_time, end_time, ...)
    - update: Update an event (event_id + fields to change)
    - delete: Delete an event (event_id)
    - freebusy: Get busy intervals (time_min, time_max, attendees)
    - auth: Force re-authentication
    - status: Check authentication status
    """
    action = parameters.get("action", "list").lower()

    if player:
        player.write_log(f"[calendar_teams] Action: {action}")

    try:
        if action == "auth":
            force_reauth()
            get_graph_token()
            return "✅ Re-authentication complete. Microsoft Graph access granted."

        elif action == "status":
            if not get_ms_client_id():
                return "❌ Not configured. Add ms_client_id (and ms_tenant_id) to config/api_keys.json."
            if is_authenticated():
                return "✅ Microsoft Graph authenticated and ready."
            return "❌ Not authenticated. Run with action='auth' to set up."

        elif action == "list":
            events = list_events(
                time_min=parameters.get("time_min"),
                time_max=parameters.get("time_max"),
                max_results=int(parameters.get("max_results", DEFAULT_MAX_RESULTS)),
                query=parameters.get("query"),
            )
            if not events:
                return "📭 No upcoming events found."

            lines = [f"📅 {len(events)} upcoming event(s):\n"]
            for i, e in enumerate(events, 1):
                lines.append(f"{i}. **{e.summary}** — {e.start} → {e.end}")
                if e.location:
                    lines.append(f"   📍 {e.location}")
                if e.meet_link:
                    lines.append(f"   🎥 {e.meet_link}")
                lines.append(f"   (id: {e.id})")
                lines.append("")
            return "\n".join(lines)

        elif action == "create":
            summary = parameters.get("summary")
            start_time = parameters.get("start_time")
            end_time = parameters.get("end_time")
            if not summary or not start_time or not end_time:
                return "❌ Please provide summary, start_time, and end_time to create an event."

            event = create_event(
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                description=parameters.get("description", ""),
                location=parameters.get("location", ""),
                attendees=parameters.get("attendees"),
                is_teams_meeting=bool(parameters.get("is_teams_meeting", False)),
            )
            lines = [f"✅ Event created: **{event.summary}**", f"   {event.start} → {event.end}"]
            if event.meet_link:
                lines.append(f"   🎥 {event.meet_link}")
            if event.html_link:
                lines.append(f"   {event.html_link}")
            return "\n".join(lines)

        elif action == "update":
            event_id = parameters.get("event_id")
            if not event_id:
                return "❌ Please provide event_id for update action."

            event = update_event(
                event_id=event_id,
                summary=parameters.get("summary"),
                start_time=parameters.get("start_time"),
                end_time=parameters.get("end_time"),
                description=parameters.get("description"),
                location=parameters.get("location"),
                attendees=parameters.get("attendees"),
            )
            return f"✅ Event updated: **{event.summary}** — {event.start} → {event.end}"

        elif action == "delete":
            event_id = parameters.get("event_id")
            if not event_id:
                return "❌ Please provide event_id for delete action."
            ok = delete_event(event_id)
            return "✅ Event deleted." if ok else "⚠️ Event not found (may already be deleted)."

        elif action == "freebusy":
            time_min = parameters.get("time_min")
            time_max = parameters.get("time_max")
            if not time_min or not time_max:
                return "❌ Please provide time_min and time_max for freebusy action."
            schedules = get_free_busy(time_min, time_max, attendees=parameters.get("attendees"))
            if not any(schedules.values()):
                return f"✅ Everyone is free for the entire window {time_min} → {time_max}."
            lines = ["🔴 Busy intervals:"]
            for email, busy in schedules.items():
                if not busy:
                    lines.append(f"   {email}: free")
                    continue
                lines.append(f"   {email}:")
                for b in busy:
                    lines.append(f"      {b['start']} → {b['end']}")
            return "\n".join(lines)

        else:
            return f"❌ Unknown action: {action}. Use: list, create, update, delete, freebusy, auth, status"

    except ConfigurationError as e:
        return f"❌ {e}"
    except TokenExpiredError:
        return ("❌ Authentication expired. Please run calendar_teams action with action='auth' "
                "to re-authenticate with Microsoft.")
    except EventNotFoundError:
        return "❌ Event not found. It may have been deleted already."
    except AuthenticationError as e:
        return f"❌ Authentication error: {e}"
    except CalendarTeamsError as e:
        return f"❌ Calendar error: {e}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Unexpected error: {e}"
# ───────────────────────────────────────────────────────────────────────────────
# Tool Declaration for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

CALENDAR_TEAMS_TOOL_DECLARATION = {
    "name": "calendar_teams",
    "description": (
        "Manages Microsoft 365 / Outlook Calendar via Graph API: list upcoming events, "
        "create/update/delete events, check free/busy time for yourself or attendees. "
        "Set is_teams_meeting=true when creating an event to attach an auto-generated "
        "Microsoft Teams join link. Use start_time/end_time as ISO 8601 datetimes with a "
        "UTC offset (e.g. '2026-09-22T15:00:00+05:30'). Actions: list, create, update, "
        "delete, freebusy, auth (re-authenticate), status (check auth)."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "list | create | update | delete | freebusy | auth | status",
            },
            "summary": {"type": "STRING", "description": "Event title (create/update)"},
            "start_time": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (create/update)"},
            "end_time": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (create/update)"},
            "description": {"type": "STRING", "description": "Event notes/body (create/update)"},
            "location": {"type": "STRING", "description": "Event location or meeting room (create/update)"},
            "attendees": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Attendee email addresses (create/update/freebusy)",
            },
            "is_teams_meeting": {"type": "BOOLEAN", "description": "Attach a Microsoft Teams join link (create only)"},
            "event_id": {"type": "STRING", "description": "Event ID (update/delete)"},
            "time_min": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (list/freebusy)"},
            "time_max": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (list/freebusy)"},
            "max_results": {"type": "INTEGER", "description": "Max events to list (default: 10)"},
            "query": {"type": "STRING", "description": "Free-text search filter (list only)"},
        },
        "required": ["action"],
    },
}
# ───────────────────────────────────────────────────────────────────────────────
# Module Test
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Teams Calendar Module")
    parser.add_argument("action", nargs="?", default="status",
                        choices=["list", "create", "auth", "status"])
    parser.add_argument("--summary", type=str, default="Test Event")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    args = parser.parse_args()

    params = {"action": args.action}
    if args.action == "create":
        params.update({"summary": args.summary, "start_time": args.start, "end_time": args.end})

    result = calendar_teams_action(params)
    print(result)
