"""
Google Calendar Module for JARVIS Assistant

API-driven calendar module for Google Calendar with OAuth 2.0 authentication.
Supports listing, creating, updating, and deleting events, plus free/busy lookups.
Optionally attaches a Google Meet link when creating an event.

Setup:
1. Reuses the same OAuth client as the Email module (config/gmail_credentials.json).
   If you have not set up Gmail yet, follow the Gmail setup steps in readme.md first.
2. In Google Cloud Console, enable the "Google Calendar API" on the same project
   (APIs & Services > Library > Google Calendar API > Enable).
3. On first use, JARVIS will open a browser for OAuth consent (this is a separate
   consent from Gmail because the scopes differ) and store tokens in
   config/gcal_token.json.
"""

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from memory.config_manager import CONFIG_DIR

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

# Full read/write access — needed for create/update/delete
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Reuses the Gmail OAuth client (same Google Cloud project, different scope/token)
CREDENTIALS_FILE = CONFIG_DIR / "gmail_credentials.json"
TOKEN_FILE = CONFIG_DIR / "gcal_token.json"

MAX_RETRIES = 3
RETRY_BACKOFF = 2
RATE_LIMIT_DELAY = 1.0

DEFAULT_MAX_RESULTS = 10
DEFAULT_CALENDAR_ID = "primary"
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
    start: str          # ISO 8601, as returned by Google
    end: str             # ISO 8601, as returned by Google
    attendees: list[str] = field(default_factory=list)
    organizer: str = ""
    html_link: str = ""
    meet_link: str = ""
    status: str = ""
    is_all_day: bool = False
# ───────────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────────

class CalendarError(Exception):
    """Base exception for calendar module errors."""
    pass
class AuthenticationError(CalendarError):
    """Authentication/authorization failure."""
    pass
class TokenExpiredError(AuthenticationError):
    """OAuth token expired and refresh failed."""
    pass
class EventNotFoundError(CalendarError):
    """Requested event does not exist."""
    pass
# ───────────────────────────────────────────────────────────────────────────────
# Token Storage
# ───────────────────────────────────────────────────────────────────────────────

def _load_token() -> Optional[Credentials]:
    """Load OAuth credentials from secure token file."""
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        return Credentials.from_authorized_user_info(data, CALENDAR_SCOPES)
    except Exception as e:
        print(f"[Calendar] ⚠️ Failed to load token: {e}")
        return None
def _save_token(creds: Credentials) -> None:
    """Save OAuth credentials to secure token file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    TOKEN_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        TOKEN_FILE.chmod(0o600)
    except Exception:
        pass  # Windows ignores chmod
def _delete_token() -> None:
    """Delete stored token (force re-auth)."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
# ───────────────────────────────────────────────────────────────────────────────
# Authentication
# ───────────────────────────────────────────────────────────────────────────────

def get_calendar_service() -> Any:
    """
    Get authenticated Google Calendar API service.
    Handles OAuth flow, token refresh, and secure storage.
    """
    creds = _load_token()

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("[Calendar] 🔄 Refreshing expired token...")
                creds.refresh(Request())
                _save_token(creds)
                print("[Calendar] ✅ Token refreshed")
            except Exception as e:
                print(f"[Calendar] ❌ Token refresh failed: {e}")
                _delete_token()
                creds = None

        if not creds:
            creds = _run_oauth_flow()
            if not creds:
                raise AuthenticationError("OAuth flow failed or was cancelled")
            _save_token(creds)
            print("[Calendar] ✅ New credentials saved")

    return build("calendar", "v3", credentials=creds, cache_discovery=False)
def _run_oauth_flow() -> Optional[Credentials]:
    """Run OAuth 2.0 authorization flow for desktop app."""
    if not CREDENTIALS_FILE.exists():
        raise AuthenticationError(
            f"Google OAuth credentials not found at {CREDENTIALS_FILE}.\n"
            "Set up Gmail credentials first (see readme.md) — Calendar reuses the "
            "same OAuth client, just save the downloaded file as config/gmail_credentials.json."
        )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            CALENDAR_SCOPES,
        )
        # Different port than the Email module so both can re-auth independently
        creds = flow.run_local_server(port=8081, access_type="offline", prompt="consent")
        return creds
    except Exception as e:
        print(f"[Calendar] ❌ OAuth flow failed: {e}")
        traceback.print_exc()
        return None
def force_reauth() -> None:
    """Force re-authentication by deleting stored token."""
    _delete_token()
    print("[Calendar] 🔐 Token deleted — next operation will require re-authentication")
# ───────────────────────────────────────────────────────────────────────────────
# API Operations with Retry Logic
# ───────────────────────────────────────────────────────────────────────────────

def _execute_with_retry(request, retries: int = MAX_RETRIES) -> Any:
    """Execute API request with exponential backoff for rate limits and network errors."""
    for attempt in range(retries):
        try:
            return request.execute()
        except HttpError as e:
            status = e.resp.status
            if status == 429:
                wait = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
                print(f"[Calendar] ⏳ Rate limited (429), waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            elif status in (500, 502, 503, 504):
                wait = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
                print(f"[Calendar] ⏳ Server error {status}, waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            elif status == 401:
                raise TokenExpiredError("Authentication token expired or invalid")
            elif status == 403:
                raise AuthenticationError(f"Insufficient permissions: {e}")
            elif status == 404:
                raise EventNotFoundError("Event not found")
            else:
                raise CalendarError(f"Calendar API error ({status}): {e}")
        except (ConnectionError, TimeoutError) as e:
            wait = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
            print(f"[Calendar] ⏳ Network error: {e}, waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        except (TokenExpiredError, AuthenticationError, EventNotFoundError):
            raise
        except Exception as e:
            raise CalendarError(f"Unexpected error: {e}")

    raise CalendarError(f"Max retries ({retries}) exceeded")
# ───────────────────────────────────────────────────────────────────────────────
# Parsing Utilities
# ───────────────────────────────────────────────────────────────────────────────

def _parse_event(item: dict) -> CalendarEvent:
    """Parse a Google Calendar API event resource into a CalendarEvent."""
    start_info = item.get("start", {})
    end_info = item.get("end", {})
    is_all_day = "date" in start_info and "dateTime" not in start_info

    meet_link = ""
    conf_data = item.get("conferenceData", {})
    for ep in conf_data.get("entryPoints", []):
        if ep.get("entryPointType") == "video":
            meet_link = ep.get("uri", "")
            break

    return CalendarEvent(
        id=item.get("id", ""),
        summary=item.get("summary", "(No title)"),
        description=item.get("description", ""),
        location=item.get("location", ""),
        start=start_info.get("dateTime") or start_info.get("date", ""),
        end=end_info.get("dateTime") or end_info.get("date", ""),
        attendees=[a.get("email", "") for a in item.get("attendees", [])],
        organizer=item.get("organizer", {}).get("email", ""),
        html_link=item.get("htmlLink", ""),
        meet_link=meet_link,
        status=item.get("status", ""),
        is_all_day=is_all_day,
    )
# ───────────────────────────────────────────────────────────────────────────────
# Core Calendar Operations
# ───────────────────────────────────────────────────────────────────────────────

def list_events(
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    query: Optional[str] = None,
) -> list[CalendarEvent]:
    """
    List upcoming events.

    Args:
        time_min: ISO 8601 datetime with UTC offset (default: now)
        time_max: ISO 8601 datetime with UTC offset (default: 7 days from time_min)
        max_results: Maximum events to return
        calendar_id: Calendar to query (default: "primary")
        query: Free-text search filter (matches summary/description/location)

    Returns:
        List of CalendarEvent objects sorted by start time
    """
    service = get_calendar_service()

    if time_min is None:
        time_min = datetime.now().astimezone().isoformat()
    if time_max is None:
        time_max = (datetime.now().astimezone() + timedelta(days=7)).isoformat()

    print(f"[Calendar] 🔍 Listing events: {time_min} → {time_max}, max={max_results}")

    req = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
        q=query,
    )
    response = _execute_with_retry(req)
    items = response.get("items", [])
    events = [_parse_event(item) for item in items]
    print(f"[Calendar] ✅ Found {len(events)} event(s)")
    return events
def create_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    attendees: Optional[list[str]] = None,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    add_meet_link: bool = False,
) -> CalendarEvent:
    """
    Create a calendar event.

    Args:
        summary: Event title
        start_time: ISO 8601 datetime with UTC offset, e.g. "2026-09-19T15:00:00+05:30"
        end_time: ISO 8601 datetime with UTC offset
        description: Event body/notes
        location: Physical location or meeting room
        attendees: List of attendee email addresses
        calendar_id: Calendar to create the event on (default: "primary")
        add_meet_link: If True, attaches an auto-generated Google Meet link

    Returns:
        The created CalendarEvent
    """
    service = get_calendar_service()

    body: dict = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]

    request_kwargs = {"calendarId": calendar_id, "body": body, "sendUpdates": "all" if attendees else "none"}

    if add_meet_link:
        body["conferenceData"] = {
            "createRequest": {
                "requestId": str(uuid.uuid4()),
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        }
        request_kwargs["conferenceDataVersion"] = 1

    req = service.events().insert(**request_kwargs)
    item = _execute_with_retry(req)
    event = _parse_event(item)
    print(f"[Calendar] ✅ Created event '{event.summary}' ({event.id})")
    return event
def update_event(
    event_id: str,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    summary: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[list[str]] = None,
) -> CalendarEvent:
    """Update an existing event. Only non-None fields are changed."""
    service = get_calendar_service()

    get_req = service.events().get(calendarId=calendar_id, eventId=event_id)
    item = _execute_with_retry(get_req)

    if summary is not None:
        item["summary"] = summary
    if description is not None:
        item["description"] = description
    if location is not None:
        item["location"] = location
    if start_time is not None:
        item["start"] = {"dateTime": start_time}
    if end_time is not None:
        item["end"] = {"dateTime": end_time}
    if attendees is not None:
        item["attendees"] = [{"email": a} for a in attendees]

    req = service.events().update(
        calendarId=calendar_id,
        eventId=event_id,
        body=item,
        sendUpdates="all" if item.get("attendees") else "none",
    )
    updated = _execute_with_retry(req)
    event = _parse_event(updated)
    print(f"[Calendar] ✅ Updated event '{event.summary}' ({event.id})")
    return event
def delete_event(event_id: str, calendar_id: str = DEFAULT_CALENDAR_ID) -> bool:
    """Delete an event by ID."""
    service = get_calendar_service()
    req = service.events().delete(calendarId=calendar_id, eventId=event_id, sendUpdates="all")
    try:
        _execute_with_retry(req)
        print(f"[Calendar] ✅ Deleted event {event_id}")
        return True
    except EventNotFoundError:
        print(f"[Calendar] ⚠️ Event {event_id} not found (already deleted?)")
        return False
def get_free_busy(
    time_min: str,
    time_max: str,
    calendar_id: str = DEFAULT_CALENDAR_ID,
) -> list[dict]:
    """
    Return busy time ranges for the given calendar between time_min and time_max.

    Returns:
        List of {"start": iso, "end": iso} busy intervals
    """
    service = get_calendar_service()
    req = service.freebusy().query(body={
        "timeMin": time_min,
        "timeMax": time_max,
        "items": [{"id": calendar_id}],
    })
    response = _execute_with_retry(req)
    busy = response.get("calendars", {}).get(calendar_id, {}).get("busy", [])
    return busy
# ───────────────────────────────────────────────────────────────────────────────
# Tool Interface for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

def calendar_google_action(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """
    JARVIS tool entry point for Google Calendar operations.

    Supported actions:
    - list: List upcoming events (time_min/time_max/max_results/query)
    - create: Create an event (summary, start_time, end_time, ...)
    - update: Update an event (event_id + fields to change)
    - delete: Delete an event (event_id)
    - freebusy: Get busy intervals (time_min, time_max)
    - auth: Force re-authentication
    - status: Check authentication status
    """
    action = parameters.get("action", "list").lower()

    if player:
        player.write_log(f"[calendar_google] Action: {action}")

    try:
        if action == "auth":
            force_reauth()
            get_calendar_service()
            return "✅ Re-authentication complete. Google Calendar access granted."

        elif action == "status":
            creds = _load_token()
            if creds and creds.valid:
                expiry = creds.expiry.strftime("%Y-%m-%d %H:%M") if creds.expiry else "unknown"
                return f"✅ Google Calendar authenticated (token expires: {expiry})"
            elif creds:
                return "⚠️ Google Calendar token exists but expired — will auto-refresh on next use"
            else:
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
                add_meet_link=bool(parameters.get("add_meet_link", False)),
            )
            lines = [f"✅ Event created: **{event.summary}**", f"   {event.start} → {event.end}"]
            if event.meet_link:
                lines.append(f"   🎥 {event.meet_link}")
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
            busy = get_free_busy(time_min, time_max)
            if not busy:
                return f"✅ Free the entire window {time_min} → {time_max}."
            lines = ["🔴 Busy intervals:"]
            for b in busy:
                lines.append(f"   {b['start']} → {b['end']}")
            return "\n".join(lines)

        else:
            return f"❌ Unknown action: {action}. Use: list, create, update, delete, freebusy, auth, status"

    except TokenExpiredError:
        return ("❌ Authentication expired. Please run calendar_google action with action='auth' "
                "to re-authenticate with Google.")
    except EventNotFoundError:
        return "❌ Event not found. It may have been deleted already."
    except AuthenticationError as e:
        return f"❌ Authentication error: {e}"
    except CalendarError as e:
        return f"❌ Calendar error: {e}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Unexpected error: {e}"
# ───────────────────────────────────────────────────────────────────────────────
# Tool Declaration for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

CALENDAR_GOOGLE_TOOL_DECLARATION = {
    "name": "calendar_google",
    "description": (
        "Manages Google Calendar via API: list upcoming events, create/update/delete events, "
        "check free/busy time. Use start_time/end_time as ISO 8601 datetimes with a UTC offset "
        "(e.g. '2026-09-19T15:00:00+05:30'). Actions: list, create, update, delete, freebusy, "
        "auth (re-authenticate), status (check auth)."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "list | create | update | delete | freebusy | auth | status",
                "enum": ["list", "create", "update", "delete", "freebusy", "auth", "status"],
            },
            "summary": {"type": "STRING", "description": "Event title (create/update)"},
            "start_time": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (create/update)"},
            "end_time": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (create/update)"},
            "description": {"type": "STRING", "description": "Event notes/body (create/update)"},
            "location": {"type": "STRING", "description": "Event location or meeting room (create/update)"},
            "attendees": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Attendee email addresses (create/update)",
            },
            "add_meet_link": {"type": "BOOLEAN", "description": "Attach a Google Meet link (create only)"},
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

    parser = argparse.ArgumentParser(description="JARVIS Google Calendar Module")
    parser.add_argument("action", nargs="?", default="status",
                        choices=["list", "create", "auth", "status"])
    parser.add_argument("--summary", type=str, default="Test Event")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    args = parser.parse_args()

    params = {"action": args.action}
    if args.action == "create":
        params.update({"summary": args.summary, "start_time": args.start, "end_time": args.end})

    result = calendar_google_action(params)
    print(result)
