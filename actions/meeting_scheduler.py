"""
Unified Meeting Scheduler for JARVIS Assistant

Sits on top of calendar_google.py and calendar_teams.py the same way email.py's
process_unread_emails() sits on top of fetch_unread_emails(): it adds a "smart"
layer (provider selection, free-slot search, merged views) over the two low-level
provider modules, instead of duplicating calendar API logic.

- If the caller gives an explicit start_time, the event is created there directly.
- If not, schedule_meeting() finds the next open slot of the requested duration on
  the chosen provider's calendar within business hours.
- list_upcoming_meetings() merges both providers' calendars into one sorted view.
- Events from list are tagged with a "provider:event_id" composite ID so cancel/
  reschedule know which provider's API to call without the caller re-specifying it.

No new external API — this module only calls into calendar_google / calendar_teams.
"""

import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from typing import Optional

from actions import calendar_google
from actions import calendar_teams
from memory.config_manager import get_calendar_provider_default, get_ms_client_id

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

DEFAULT_DURATION_MIN = 30
DEFAULT_SEARCH_DAYS = 5
BUSINESS_START_HOUR = 9
BUSINESS_END_HOUR = 18
# ───────────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────────

class SchedulingError(Exception):
    """Base exception for meeting scheduler errors."""
    pass
class NoProviderConfiguredError(SchedulingError):
    """Neither Google Calendar nor Teams Calendar is set up."""
    pass
class NoFreeSlotError(SchedulingError):
    """Could not find an open slot within the search window."""
    pass
# ───────────────────────────────────────────────────────────────────────────────
# Provider Resolution
# ───────────────────────────────────────────────────────────────────────────────

def _google_configured() -> bool:
    return calendar_google.CREDENTIALS_FILE.exists()
def _teams_configured() -> bool:
    return bool(get_ms_client_id())
def available_providers() -> list[str]:
    """Providers that have credentials/app registration set up (not necessarily authenticated yet)."""
    providers = []
    if _google_configured():
        providers.append("google")
    if _teams_configured():
        providers.append("teams")
    return providers
def resolve_provider(explicit: Optional[str] = None) -> str:
    """
    Decide which calendar provider to use.

    Priority: explicit param > configured default > single configured provider > "google" tie-break.
    """
    if explicit and explicit.lower() in ("google", "teams"):
        return explicit.lower()

    providers = available_providers()
    if not providers:
        raise NoProviderConfiguredError(
            "No calendar provider is set up yet. Configure Google Calendar "
            "(config/gmail_credentials.json) or Teams Calendar (ms_client_id in "
            "config/api_keys.json) first."
        )

    default = get_calendar_provider_default()
    if default in providers:
        return default

    if len(providers) == 1:
        return providers[0]

    # Both configured, no default set — arbitrary but consistent tie-break
    return "google"
# ───────────────────────────────────────────────────────────────────────────────
# Datetime Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    """Parse a Google/Graph datetime string (various formats) into an aware UTC datetime."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "." in s:
        head, rest = s.split(".", 1)
        # split fractional seconds from any trailing offset
        tz_part = ""
        for sep in ("+", "-"):
            idx = rest.find(sep, 1)
            if idx != -1:
                tz_part = rest[idx:]
                rest = rest[:idx]
                break
        digits = "".join(ch for ch in rest if ch.isdigit())[:6]
        s = f"{head}.{digits}{tz_part}" if digits else f"{head}{tz_part}"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
def _get_busy_intervals(
    provider: str,
    time_min: str,
    time_max: str,
    attendees: Optional[list[str]] = None,
) -> list[tuple[datetime, datetime]]:
    """Return merged (start, end) busy intervals for the organizer's calendar (and,
    for Teams, any given attendees within the same tenant)."""
    if provider == "google":
        busy = calendar_google.get_free_busy(time_min, time_max)
        return [(_parse_dt(b["start"]), _parse_dt(b["end"])) for b in busy]

    schedules = calendar_teams.get_free_busy(time_min, time_max, attendees=attendees)
    intervals = []
    for items in schedules.values():
        for it in items:
            status = (it.get("status") or "free").lower()
            if status not in ("free", ""):
                intervals.append((_parse_dt(it["start"]), _parse_dt(it["end"])))
    return intervals
def find_free_slot(
    provider: str,
    duration_min: int,
    earliest_start: Optional[str] = None,
    search_days: int = DEFAULT_SEARCH_DAYS,
    attendees: Optional[list[str]] = None,
) -> Optional[tuple[str, str]]:
    """
    Find the next open slot of duration_min minutes within business hours
    (09:00-18:00 UTC) starting from earliest_start (default: now).

    Returns:
        (start_iso, end_iso) in UTC, or None if nothing found in the search window.
    """
    now = datetime.now(timezone.utc)
    cursor = _parse_dt(earliest_start) if earliest_start else now
    if cursor < now:
        cursor = now

    for day_offset in range(search_days):
        day = (cursor.date() + timedelta(days=day_offset))
        day_start = datetime.combine(day, dtime(BUSINESS_START_HOUR, 0), tzinfo=timezone.utc)
        day_end = datetime.combine(day, dtime(BUSINESS_END_HOUR, 0), tzinfo=timezone.utc)
        window_start = max(day_start, cursor) if day_offset == 0 else day_start
        if window_start >= day_end:
            continue

        busy = sorted(
            _get_busy_intervals(provider, window_start.isoformat(), day_end.isoformat(), attendees),
            key=lambda b: b[0],
        )

        probe = window_start
        for b_start, b_end in busy:
            if b_start > probe and (b_start - probe) >= timedelta(minutes=duration_min):
                return probe.isoformat(), (probe + timedelta(minutes=duration_min)).isoformat()
            if b_end > probe:
                probe = b_end
        if (day_end - probe) >= timedelta(minutes=duration_min):
            return probe.isoformat(), (probe + timedelta(minutes=duration_min)).isoformat()

    return None
# ───────────────────────────────────────────────────────────────────────────────
# Composite Event IDs ("provider:raw_id")
# ───────────────────────────────────────────────────────────────────────────────

def _make_composite_id(provider: str, raw_id: str) -> str:
    return f"{provider}:{raw_id}"
def _split_composite_id(event_id: str, fallback_provider: Optional[str] = None) -> tuple[str, str]:
    if ":" in event_id:
        provider, raw_id = event_id.split(":", 1)
        if provider in ("google", "teams"):
            return provider, raw_id
    if not fallback_provider:
        raise SchedulingError(
            f"Ambiguous event_id '{event_id}' — prefix it with 'google:' or 'teams:', "
            "or pass a provider explicitly."
        )
    return fallback_provider, event_id
# ───────────────────────────────────────────────────────────────────────────────
# Data Class
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledMeeting:
    provider: str
    composite_id: str
    summary: str
    start: str
    end: str
    location: str
    meet_link: str
    html_link: str
# ───────────────────────────────────────────────────────────────────────────────
# High-Level Operations
# ───────────────────────────────────────────────────────────────────────────────

def schedule_meeting(
    title: str,
    duration_min: int = DEFAULT_DURATION_MIN,
    attendees: Optional[list[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    earliest_start: Optional[str] = None,
    provider: Optional[str] = None,
    description: str = "",
    location: str = "",
    video_link: bool = True,
    search_days: int = DEFAULT_SEARCH_DAYS,
) -> ScheduledMeeting:
    """
    Schedule a meeting, picking a free slot automatically if start_time isn't given.

    Args:
        title: Meeting title
        duration_min: Meeting length in minutes (used for auto-slot search, or to
            derive end_time when only start_time is given)
        attendees: Attendee email addresses
        start_time: ISO 8601 datetime with UTC offset — if given, used as-is (no slot search)
        end_time: ISO 8601 datetime with UTC offset — derived from start_time + duration_min if omitted
        earliest_start: ISO 8601 datetime with UTC offset — earliest the auto-found slot may start
        provider: "google" | "teams" | None (auto-resolve)
        description: Meeting notes/agenda
        location: Physical location or meeting room
        video_link: Attach a Google Meet / Teams join link
        search_days: How many days ahead to search for a free slot

    Returns:
        ScheduledMeeting with the created event's details
    """
    resolved = resolve_provider(provider)

    if start_time and not end_time:
        end_time = (_parse_dt(start_time) + timedelta(minutes=duration_min)).astimezone(timezone.utc).isoformat()

    if not start_time:
        slot = find_free_slot(resolved, duration_min, earliest_start=earliest_start,
                               search_days=search_days, attendees=attendees)
        if not slot:
            raise NoFreeSlotError(
                f"No free {duration_min}-minute slot found on the {resolved} calendar "
                f"in the next {search_days} day(s)."
            )
        start_time, end_time = slot

    if resolved == "google":
        event = calendar_google.create_event(
            summary=title, start_time=start_time, end_time=end_time,
            description=description, location=location, attendees=attendees,
            add_meet_link=video_link,
        )
    else:
        event = calendar_teams.create_event(
            summary=title, start_time=start_time, end_time=end_time,
            description=description, location=location, attendees=attendees,
            is_teams_meeting=video_link,
        )

    return ScheduledMeeting(
        provider=resolved,
        composite_id=_make_composite_id(resolved, event.id),
        summary=event.summary,
        start=event.start,
        end=event.end,
        location=event.location,
        meet_link=event.meet_link,
        html_link=event.html_link,
    )
def list_upcoming_meetings(
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = 10,
    provider: Optional[str] = None,
) -> list[ScheduledMeeting]:
    """
    List upcoming meetings, merged across whichever providers are configured
    (or a single provider if explicitly requested), sorted by start time.
    """
    providers = [provider.lower()] if provider and provider.lower() in ("google", "teams") else available_providers()
    if not providers:
        raise NoProviderConfiguredError(
            "No calendar provider is set up yet. Configure Google Calendar or Teams Calendar first."
        )

    merged: list[ScheduledMeeting] = []
    for p in providers:
        try:
            module = calendar_google if p == "google" else calendar_teams
            events = module.list_events(time_min=time_min, time_max=time_max, max_results=max_results)
            for e in events:
                merged.append(ScheduledMeeting(
                    provider=p,
                    composite_id=_make_composite_id(p, e.id),
                    summary=e.summary,
                    start=e.start,
                    end=e.end,
                    location=e.location,
                    meet_link=e.meet_link,
                    html_link=e.html_link,
                ))
        except Exception as e:
            print(f"[MeetingScheduler] ⚠️ Failed to list {p} events: {e}")

    merged.sort(key=lambda m: _parse_dt(m.start) if m.start else datetime.max.replace(tzinfo=timezone.utc))
    return merged[:max_results]
def cancel_meeting(event_id: str, provider: Optional[str] = None) -> bool:
    """Cancel a meeting by composite ID ('google:abc123') or raw ID + explicit provider."""
    resolved_provider, raw_id = _split_composite_id(event_id, provider)
    module = calendar_google if resolved_provider == "google" else calendar_teams
    return module.delete_event(raw_id)
def reschedule_meeting(
    event_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    provider: Optional[str] = None,
) -> ScheduledMeeting:
    """Reschedule a meeting to a new start/end time by composite ID or raw ID + explicit provider."""
    resolved_provider, raw_id = _split_composite_id(event_id, provider)
    module = calendar_google if resolved_provider == "google" else calendar_teams
    event = module.update_event(raw_id, start_time=start_time, end_time=end_time)
    return ScheduledMeeting(
        provider=resolved_provider,
        composite_id=_make_composite_id(resolved_provider, event.id),
        summary=event.summary,
        start=event.start,
        end=event.end,
        location=event.location,
        meet_link=event.meet_link,
        html_link=event.html_link,
    )
# ───────────────────────────────────────────────────────────────────────────────
# Tool Interface for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

def meeting_scheduler_action(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """
    JARVIS tool entry point for unified meeting scheduling.

    Supported actions:
    - schedule: Create a meeting, auto-finding a free slot if start_time isn't given
    - list: List upcoming meetings merged across configured providers
    - cancel: Cancel a meeting (event_id as 'provider:id', or event_id + provider)
    - reschedule: Move a meeting to a new time (event_id + start_time/end_time)
    """
    action = parameters.get("action", "list").lower()

    if player:
        player.write_log(f"[meeting_scheduler] Action: {action}")

    try:
        if action == "list":
            meetings = list_upcoming_meetings(
                time_min=parameters.get("time_min"),
                time_max=parameters.get("time_max"),
                max_results=int(parameters.get("max_results", 10)),
                provider=parameters.get("provider"),
            )
            if not meetings:
                return "📭 No upcoming meetings found."

            icon = {"google": "🟦", "teams": "🟪"}
            lines = [f"📅 {len(meetings)} upcoming meeting(s):\n"]
            for i, m in enumerate(meetings, 1):
                lines.append(f"{i}. {icon.get(m.provider, '')} **{m.summary}** — {m.start} → {m.end}")
                if m.location:
                    lines.append(f"   📍 {m.location}")
                if m.meet_link:
                    lines.append(f"   🎥 {m.meet_link}")
                lines.append(f"   (id: {m.composite_id})")
                lines.append("")
            return "\n".join(lines)

        elif action == "schedule":
            title = parameters.get("title") or parameters.get("summary")
            if not title:
                return "❌ Please provide a title for the meeting."

            meeting = schedule_meeting(
                title=title,
                duration_min=int(parameters.get("duration_min", DEFAULT_DURATION_MIN)),
                attendees=parameters.get("attendees"),
                start_time=parameters.get("start_time"),
                end_time=parameters.get("end_time"),
                earliest_start=parameters.get("earliest_start"),
                provider=parameters.get("provider"),
                description=parameters.get("description", ""),
                location=parameters.get("location", ""),
                video_link=bool(parameters.get("video_link", True)),
                search_days=int(parameters.get("search_days", DEFAULT_SEARCH_DAYS)),
            )
            icon = "🟦" if meeting.provider == "google" else "🟪"
            lines = [
                f"✅ {icon} Meeting scheduled on {meeting.provider}: **{meeting.summary}**",
                f"   {meeting.start} → {meeting.end}",
            ]
            if meeting.meet_link:
                lines.append(f"   🎥 {meeting.meet_link}")
            lines.append(f"   (id: {meeting.composite_id})")
            return "\n".join(lines)

        elif action == "cancel":
            event_id = parameters.get("event_id")
            if not event_id:
                return "❌ Please provide event_id to cancel."
            ok = cancel_meeting(event_id, provider=parameters.get("provider"))
            return "✅ Meeting cancelled." if ok else "⚠️ Meeting not found (may already be cancelled)."

        elif action == "reschedule":
            event_id = parameters.get("event_id")
            if not event_id:
                return "❌ Please provide event_id to reschedule."
            if not parameters.get("start_time") and not parameters.get("end_time"):
                return "❌ Please provide a new start_time (and optionally end_time)."
            meeting = reschedule_meeting(
                event_id=event_id,
                start_time=parameters.get("start_time"),
                end_time=parameters.get("end_time"),
                provider=parameters.get("provider"),
            )
            return f"✅ Meeting rescheduled: **{meeting.summary}** — {meeting.start} → {meeting.end}"

        else:
            return f"❌ Unknown action: {action}. Use: schedule, list, cancel, reschedule"

    except NoProviderConfiguredError as e:
        return f"❌ {e}"
    except NoFreeSlotError as e:
        return f"❌ {e}"
    except SchedulingError as e:
        return f"❌ {e}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Unexpected error: {e}"
# ───────────────────────────────────────────────────────────────────────────────
# Tool Declaration for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

MEETING_SCHEDULER_TOOL_DECLARATION = {
    "name": "meeting_scheduler",
    "description": (
        "Unified meeting scheduling across Google Calendar and Teams Calendar. Prefer this "
        "tool over calling calendar_google/calendar_teams directly for natural-language "
        "requests like 'schedule a meeting with X tomorrow', 'what's on my calendar', "
        "'cancel my 3pm', or 'move my meeting to 4pm'. If start_time is omitted on 'schedule', "
        "it automatically finds the next free slot of duration_min minutes within business "
        "hours. 'list' merges events from every configured provider, each tagged with a "
        "composite id like 'google:abc123' or 'teams:xyz789' — pass that id back for cancel/"
        "reschedule. Use start_time/end_time as ISO 8601 datetimes with a UTC offset."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "schedule | list | cancel | reschedule",
            },
            "title": {"type": "STRING", "description": "Meeting title (schedule)"},
            "duration_min": {"type": "INTEGER", "description": "Meeting length in minutes (default: 30)"},
            "attendees": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Attendee email addresses (schedule)",
            },
            "start_time": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset — exact start (schedule/reschedule)"},
            "end_time": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (schedule/reschedule)"},
            "earliest_start": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset — earliest allowed auto-found start (schedule, only used when start_time is omitted)"},
            "provider": {"type": "STRING", "description": "google | teams (omit to auto-resolve)"},
            "description": {"type": "STRING", "description": "Meeting notes/agenda (schedule)"},
            "location": {"type": "STRING", "description": "Physical location or meeting room (schedule)"},
            "video_link": {"type": "BOOLEAN", "description": "Attach a video call link (default: true) (schedule)"},
            "search_days": {"type": "INTEGER", "description": "Days ahead to search for a free slot (default: 5) (schedule)"},
            "event_id": {"type": "STRING", "description": "Composite id 'provider:id' from a list result (cancel/reschedule)"},
            "time_min": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (list)"},
            "time_max": {"type": "STRING", "description": "ISO 8601 datetime with UTC offset (list)"},
            "max_results": {"type": "INTEGER", "description": "Max meetings to list (default: 10)"},
        },
        "required": ["action"],
    },
}
# ───────────────────────────────────────────────────────────────────────────────
# Module Test
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Meeting Scheduler Module")
    parser.add_argument("action", nargs="?", default="list",
                        choices=["list", "schedule", "cancel", "reschedule"])
    parser.add_argument("--title", type=str, default="Test Meeting")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()

    params = {"action": args.action, "provider": args.provider}
    if args.action == "schedule":
        params.update({"title": args.title, "duration_min": args.duration})

    result = meeting_scheduler_action(params)
    print(result)
