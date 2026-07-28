"""
Email Module for JARVIS Assistant

API-driven email module for Gmail API (Google) with OAuth 2.0 authentication.
Supports fetching unread emails, AI-powered reply generation, and draft creation.
Does NOT send emails directly — creates drafts for human review.

Setup:
1. Go to Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API: APIs & Services > Library > Gmail API > Enable
4. Create OAuth 2.0 credentials: APIs & Services > Credentials > Create Credentials > OAuth Client ID
   - Application type: Desktop Application
   - Authorized redirect URIs: http://localhost:8080/ (or use the default http://localhost)
5. Download credentials JSON and save as config/gmail_credentials.json
6. On first run, JARVIS will open a browser for OAuth consent and store tokens securely
"""

import base64
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage # type: ignore
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional, List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import BatchHttpRequest

from core.llm_client import call_llm
from memory.config_manager import (
    CONFIG_DIR,
    load_api_keys,
    get_user_email,
    get_user_context,
)

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

# Gmail API scopes — modify if you need different permissions
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",  # For creating drafts
    "https://www.googleapis.com/auth/gmail.modify",   # For marking as read
]

# File paths
CREDENTIALS_FILE = CONFIG_DIR / "gmail_credentials.json"
TOKEN_FILE = CONFIG_DIR / "gmail_token.json"

# Rate limiting
MAX_EMAILS_PER_FETCH = 20
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # exponential backoff base

# Email fetch defaults
DEFAULT_MAX_RESULTS = 10
DEFAULT_DAYS_BACK = 7
# ───────────────────────────────────────────────────────────────────────────────
# Data Classes
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class EmailMessage:
    """Clean email data structure with extracted metadata and plain-text body."""
    id: str
    thread_id: str
    sender: str
    recipient: str
    subject: str
    date: datetime
    snippet: str
    body_text: str
    is_unread: bool
    labels: list[str]
    references: str = ""
@dataclass
class DraftReply:
    """Draft reply ready for user review."""
    thread_id: str
    to: str
    subject: str
    body: str
    in_reply_to: str
    references: str
# ───────────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────────

class EmailError(Exception):
    """Base exception for email module errors."""
    pass
class AuthenticationError(EmailError):
    """Authentication/authorization failure."""
    pass
class RateLimitError(EmailError):
    """API rate limit exceeded."""
    pass
class NetworkError(EmailError):
    """Network/connectivity issues."""
    pass
class TokenExpiredError(AuthenticationError):
    """OAuth token expired and refresh failed."""
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
        creds = Credentials.from_authorized_user_info(data, GMAIL_SCOPES)
        return creds
    except Exception as e:
        print(f"[Email] ⚠️ Failed to load token: {e}")
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
    # Restrict file permissions on Unix-like systems
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

def get_gmail_service() -> Any:
    """
    Get authenticated Gmail API service.
    Handles OAuth flow, token refresh, and secure storage.
    """
    creds = _load_token()

    # No valid credentials — run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("[Email] 🔄 Refreshing expired token...")
                creds.refresh(Request())
                _save_token(creds)
                print("[Email] ✅ Token refreshed")
            except Exception as e:
                print(f"[Email] ❌ Token refresh failed: {e}")
                _delete_token()
                creds = None

        if not creds:
            creds = _run_oauth_flow()
            if not creds:
                raise AuthenticationError("OAuth flow failed or was cancelled")
            _save_token(creds)
            print("[Email] ✅ New credentials saved")

    return build("gmail", "v1", credentials=creds, cache_discovery=False)
def _run_oauth_flow() -> Optional[Credentials]:
    """Run OAuth 2.0 authorization flow for desktop app."""
    if not CREDENTIALS_FILE.exists():
        raise AuthenticationError(
            f"Gmail credentials not found at {CREDENTIALS_FILE}.\n"
            "Please download OAuth credentials from Google Cloud Console and save as "
            "config/gmail_credentials.json. See module docstring for setup instructions."
        )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            GMAIL_SCOPES,
        )
        # Run local server on port 8080 for OAuth callback
        creds = flow.run_local_server(port=8080, access_type="offline", prompt="consent")
        return creds
    except Exception as e:
        print(f"[Email] ❌ OAuth flow failed: {e}")
        traceback.print_exc()
        return None
def force_reauth() -> None:
    """Force re-authentication by deleting stored token."""
    _delete_token()
    print("[Email] 🔐 Token deleted — next operation will require re-authentication")
# ───────────────────────────────────────────────────────────────────────────────
# Email Parsing Utilities
# ───────────────────────────────────────────────────────────────────────────────

def _decode_base64url(data: str) -> bytes:
    """Decode base64url-encoded string (Gmail API format)."""
    # Add padding if needed
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)
def _extract_body(payload: dict) -> str:
    """
    Recursively extract plain-text body from Gmail message payload.
    Prefers text/plain over text/html, strips HTML if only HTML available.
    """
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data", "")

    # Leaf node with content
    if data:
        try:
            decoded = _decode_base64url(data).decode("utf-8", errors="replace")
            if mime_type == "text/plain":
                return decoded
            elif mime_type == "text/html":
                return _strip_html(decoded)
        except Exception:
            pass

    # Multipart — recurse into parts
    parts = payload.get("parts", [])
    for part in parts:
        part_text = _extract_body(part)
        if part_text:
            # Prefer plain text
            if part.get("mimeType") == "text/plain":
                return part_text

    # Fallback: return first non-empty part (likely HTML)
    for part in parts:
        part_text = _extract_body(part)
        if part_text:
            return part_text

    return ""
def _strip_html(html: str) -> str:
    """Convert HTML to clean plain text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "head", "meta", "link", "noscript"]):
            tag.decompose()

        # Get text with reasonable formatting
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)
    except Exception:
        # Fallback: crude tag stripping
        import re
        text = re.sub(r"<[^>]+>", "", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
def _parse_email_date(date_str: str) -> datetime:
    """Parse Gmail date header to datetime."""
    # Gmail format: "Wed, 15 Jan 2025 14:30:00 +0000"
    try:
        return datetime.strptime(date_str[:31], "%a, %d %b %Y %H:%M:%S")
    except Exception:
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now()
def _extract_header(headers: list[dict], name: str) -> str:
    """Extract header value by name (case-insensitive)."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""
def _parse_gmail_message(msg: dict) -> EmailMessage:
    """Parse Gmail API message response into EmailMessage."""
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    # Extract headers
    subject = _extract_header(headers, "Subject")
    sender = _extract_header(headers, "From")
    recipient = _extract_header(headers, "To")
    date_str = _extract_header(headers, "Date")
    message_id = _extract_header(headers, "Message-ID")
    references = _extract_header(headers, "References")

    # Parse date
    date = _parse_email_date(date_str)

    # Extract body
    body_text = _extract_body(payload)

    # Labels and unread status
    labels = msg.get("labelIds", [])
    is_unread = "UNREAD" in labels

    return EmailMessage(
        id=msg.get("id", ""),
        thread_id=msg.get("threadId", ""),
        sender=sender,
        recipient=recipient,
        subject=subject or "(No Subject)",
        date=date,
        snippet=msg.get("snippet", ""),
        body_text=body_text.strip(),
        is_unread=is_unread,
        labels=labels,
        references=references,
    )
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
            if status == 429:  # Rate limit
                wait = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
                print(f"[Email] ⏳ Rate limited (429), waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            elif status in (500, 502, 503, 504):  # Server errors
                wait = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
                print(f"[Email] ⏳ Server error {status}, waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            elif status == 401:  # Unauthorized — token expired
                raise TokenExpiredError("Authentication token expired or invalid")
            elif status == 403:  # Forbidden — insufficient permissions
                raise AuthenticationError(f"Insufficient permissions: {e}")
            else:
                raise EmailError(f"Gmail API error ({status}): {e}")
        except (ConnectionError, TimeoutError) as e:
            wait = RATE_LIMIT_DELAY * (RETRY_BACKOFF ** attempt)
            print(f"[Email] ⏳ Network error: {e}, waiting {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        except Exception as e:
            raise EmailError(f"Unexpected error: {e}")

    raise EmailError(f"Max retries ({retries}) exceeded")
# ───────────────────────────────────────────────────────────────────────────────
# Core Email Operations
# ───────────────────────────────────────────────────────────────────────────────

def fetch_unread_emails(
    max_results: int = DEFAULT_MAX_RESULTS,
    days_back: int = DEFAULT_DAYS_BACK,
    include_read: bool = False,
    query: Optional[str] = None,
) -> list[EmailMessage]:
    """
    Fetch recent emails from Gmail.

    Args:
        max_results: Maximum number of emails to fetch (default: 10, max: 100)
        days_back: How many days back to search (default: 7)
        include_read: If True, include read emails; if False, only unread
        query: Custom Gmail search query (overrides other filters)

    Returns:
        List of EmailMessage objects sorted newest first
    """
    service = get_gmail_service()

    # Build query
    if query is None:
        query_parts = []
        if not include_read:
            query_parts.append("is:unread")
        if days_back > 0:
            since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
            query_parts.append(f"after:{since_date}")
        query = " ".join(query_parts) if query_parts else ""

    print(f"[Email] 🔍 Fetching emails: query='{query}', max={max_results}")

    # List messages
    try:
        list_req = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=min(max_results, MAX_EMAILS_PER_FETCH),
        )
        response = _execute_with_retry(list_req)
    except HttpError as e:
        if e.resp.status == 401:
            raise TokenExpiredError("Authentication failed — token may be expired")
        raise

    messages = response.get("messages", [])
    if not messages:
        print("[Email] 📭 No messages found")
        return []

    print(f"[Email] 📬 Found {len(messages)} messages, fetching details...")

    # Batch fetch full messages
    emails = []
    for i, msg_ref in enumerate(messages):
        try:
            get_req = service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="full",
            )
            msg = _execute_with_retry(get_req)
            email = _parse_gmail_message(msg)
            emails.append(email)

            # Small delay to respect rate limits
            if i < len(messages) - 1:
                time.sleep(RATE_LIMIT_DELAY)

        except TokenExpiredError:
            raise
        except Exception as e:
            print(f"[Email] ⚠️ Failed to fetch message {msg_ref['id']}: {e}")
            continue

    # Sort newest first
    emails.sort(key=lambda e: e.date, reverse=True)
    print(f"[Email] ✅ Fetched {len(emails)} emails")
    return emails
def mark_as_read(email_ids: list[str]) -> bool:
    """Mark emails as read by removing UNREAD label."""
    if not email_ids:
        return True

    service = get_gmail_service()
    try:
        batch = service.new_batch_http_request()
        for msg_id in email_ids:
            batch.add(
                service.users().messages().modify(
                    userId="me",
                    id=msg_id,
                    body={"removeLabelIds": ["UNREAD"]},
                )
            )
        batch.execute()
        print(f"[Email] ✅ Marked {len(email_ids)} emails as read")
        return True
    except Exception as e:
        print(f"[Email] ⚠️ Failed to mark as read: {e}")
        return False
# ───────────────────────────────────────────────────────────────────────────────
# AI Reply Generation
# ───────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_EMAIL_REPLY = """You are JARVIS, an AI assistant drafting email replies for your user.
Your task: Write a concise, professional, context-aware reply to the incoming email.

Guidelines:
- Match the tone of the incoming email (formal/casual, brief/detailed)
- Address the sender by name if available
- Answer questions directly; ask clarifying questions if needed
- Keep replies concise — 2-5 sentences typically
- Do NOT make up information; if unsure, say "I'll check and get back to you"
- Sign off as the user would (use their name from context if known)
- Output ONLY the reply body — no greetings like "Here's a draft:" or explanations

Context about the user:
- Name: {user_name}
- Role/Context: {user_context}
"""
def generate_ai_reply(
    email: EmailMessage,
    user_name: str = "",
    user_context: str = "",
    custom_instructions: str = "",
) -> str:
    """
    Generate a contextual AI reply for the given email.

    Args:
        email: The email to reply to
        user_name: Your name for signing off
        user_context: Brief context about you (role, projects, etc.)
        custom_instructions: Additional instructions for the AI

    Returns:
        Plain-text reply body ready for draft creation
    """
    # Truncate very long emails to fit context window
    max_body_len = 30000
    body = email.body_text
    if len(body) > max_body_len:
        body = body[:max_body_len] + "\n\n[...truncated...]"

    # Build prompt
    prompt = f"""Incoming email:
From: {email.sender}
To: {email.recipient}
Subject: {email.subject}
Date: {email.date.strftime('%Y-%m-%d %H:%M')}
---
{body}

---
{custom_instructions}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EMAIL_REPLY.format(
            user_name=user_name or "the user",
            user_context=user_context or "General professional",
        )},
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_llm(messages, tools=None, timeout=60)
        reply = response.get("content", "").strip()

        # Clean up any AI meta-commentary
        if reply.lower().startswith(("here is", "here's a", "draft:", "reply:")):
            lines = reply.split("\n")
            reply = "\n".join(lines[1:]).strip()

        return reply
    except Exception as e:
        print(f"[Email] ❌ AI reply generation failed: {e}")
        raise EmailError(f"Failed to generate AI reply: {e}")
# ───────────────────────────────────────────────────────────────────────────────
# Email Summarization
# ───────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_EMAIL_SUMMARY = """You are JARVIS, an AI assistant that summarizes emails concisely.
Your task: Write a brief, accurate summary of the email content.

Guidelines:
- Identify the key topic, request, or action item
- Mention important dates, deadlines, or decisions
- Keep it to 1-3 sentences
- Do NOT include meta-commentary or greetings
- Output ONLY the summary text
"""

def summarize_email(
    email: EmailMessage,
    custom_instructions: str = "",
) -> str:
    """
    Generate a concise AI summary of an email.

    Args:
        email: The email to summarize
        custom_instructions: Additional instructions for the AI

    Returns:
        Plain-text summary (1-3 sentences)
    """
    max_body_len = 30000
    body = email.body_text
    if len(body) > max_body_len:
        body = body[:max_body_len] + "\n\n[...truncated...]"

    prompt = f"""Email to summarize:
From: {email.sender}
To: {email.recipient}
Subject: {email.subject}
Date: {email.date.strftime('%Y-%m-%d %H:%M')}
---
{body}

---
{custom_instructions}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EMAIL_SUMMARY},
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_llm(messages, tools=None, timeout=60)
        summary = response.get("content", "").strip()

        # Clean up any AI meta-commentary
        if summary.lower().startswith(("here is", "here's a", "summary:", "summary:")):
            lines = summary.split("\n")
            summary = "\n".join(lines[1:]).strip()

        return summary
    except Exception as e:
        print(f"[Email] ❌ AI summary generation failed: {e}")
        raise EmailError(f"Failed to generate email summary: {e}")
# ───────────────────────────────────────────────────────────────────────────────
# Draft Creation
# ───────────────────────────────────────────────────────────────────────────────

def create_draft_reply(
    email: EmailMessage,
    reply_body: str,
    user_email: str = "",
) -> DraftReply:
    """
    Create a draft reply in Gmail (does NOT send).

    Args:
        email: Original email to reply to
        reply_body: Plain-text reply content
        user_email: Your email address (for From header)

    Returns:
        DraftReply object with draft details
    """
    service = get_gmail_service()

    # Build reply message (use MIMEText to create proper email message)
    message = MIMEText(reply_body)
    message["To"] = email.sender
    message["Subject"] = f"Re: {email.subject}" if not email.subject.lower().startswith("re:") else email.subject
    message["In-Reply-To"] = email.id
    message["References"] = f"{email.references} {email.id}".strip() if email.references else email.id

    if user_email:
        message["From"] = user_email

    # Encode for Gmail API
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    # Create draft
    draft_body = {
        "message": {
            "raw": raw,
            "threadId": email.thread_id,
        }
    }

    try:
        req = service.users().drafts().create(userId="me", body=draft_body)
        draft = _execute_with_retry(req)

        print(f"[Email] ✅ Draft created for thread {email.thread_id}")
        return DraftReply(
            thread_id=email.thread_id,
            to=email.sender,
            subject=message["Subject"],
            body=reply_body,
            in_reply_to=email.id,
            references=message["References"],
        )
    except HttpError as e:
        if e.resp.status == 401:
            raise TokenExpiredError("Authentication failed creating draft")
        raise EmailError(f"Failed to create draft: {e}")
# ───────────────────────────────────────────────────────────────────────────────
# Email Sending
# ───────────────────────────────────────────────────────────────────────────────

def send_email(
    to: str,
    subject: str,
    body: str,
    user_email: str = "",
    thread_id: str = "",
    in_reply_to: str = "",
    references: str = "",
) -> dict:
    """
    Send an email immediately via Gmail API.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Plain-text email body
        user_email: Your email address (for From header)
        thread_id: Optional thread ID to send as reply in existing thread
        in_reply_to: Optional Message-ID this email replies to
        references: Optional References header for threading

    Returns:
        Dict with message_id, thread_id, and status
    """
    service = get_gmail_service()

    # Build the email message
    message = MIMEText(body)
    message["To"] = to
    message["Subject"] = subject
    if user_email:
        message["From"] = user_email
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
    if references:
        message["References"] = references

    # Encode for Gmail API
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    # Build the send request body
    send_body: dict[str, Any] = {"raw": raw}
    if thread_id:
        send_body["threadId"] = thread_id

    try:
        req = service.users().messages().send(userId="me", body=send_body)
        sent = _execute_with_retry(req)

        message_id = sent.get("id", "")
        result_thread_id = sent.get("threadId", thread_id or "")

        print(f"[Email] ✅ Email sent to {to} (message ID: {message_id})")
        return {
            "status": "sent",
            "message_id": message_id,
            "thread_id": result_thread_id,
            "to": to,
            "subject": subject,
        }
    except HttpError as e:
        if e.resp.status == 401:
            raise TokenExpiredError("Authentication failed sending email")
        elif e.resp.status == 403:
            raise AuthenticationError(f"Insufficient permissions to send: {e}")
        raise EmailError(f"Failed to send email: {e}")
# ───────────────────────────────────────────────────────────────────────────────
# High-Level Workflow
# ───────────────────────────────────────────────────────────────────────────────

def process_unread_emails(
    max_results: int = DEFAULT_MAX_RESULTS,
    days_back: int = DEFAULT_DAYS_BACK,
    include_read: bool = False,
    query: Optional[str] = None,
    user_name: str = "",
    user_context: str = "",
    user_email: str = "",
    custom_instructions: str = "",
    mark_read_after: bool = True,
    target_email_ids: Optional[List[str]] = None,
) -> list[dict]:
    """
    Complete workflow: fetch emails → generate AI replies → create drafts.

    Args:
        max_results: Max emails to process
        days_back: How many days back to search
        include_read: Include read emails (default: false, only unread)
        query: Custom Gmail search query
        user_name: Your name for sign-off
        user_context: Brief context about you
        user_email: Your email address
        custom_instructions: Additional AI instructions
        mark_read_after: Mark processed emails as read
        target_email_ids: List of specific email IDs to create drafts for (only these will be processed)

    Returns:
        List of result dicts with email info and draft status
    """
    results = []

    # Load user config if not provided
    if not user_name or not user_email:
        config = load_api_keys()
        user_name = user_name or config.get("user_name", "")
        user_email = user_email or config.get("user_email", "")

    try:
        emails = fetch_unread_emails(
            max_results=max_results,
            days_back=days_back,
            include_read=include_read,
            query=query,
        )
    except TokenExpiredError:
        raise
    except Exception as e:
        raise EmailError(f"Failed to fetch emails: {e}")

    # Strict validation - filter emails if target_email_ids is provided
    if target_email_ids is not None:
        # Create a set for faster lookup
        target_ids_set = set(target_email_ids)

        # Filter emails to only those in the target list
        emails = [email for email in emails if email.id in target_ids_set]

        # Log the filtering activity
        print(f"[Email] 🎯 Filtering: Found {len(emails)} matching emails out of {len(target_email_ids)} requested")
        if emails:
            print(f"[Email] 📧 Matching emails: {', '.join([e.subject for e in emails])}")

    # No emails to process after filtering
    if not emails:
        if target_email_ids is not None:
            return [{"status": "no_matches", "message": f"No matching emails found. Requested IDs: {', '.join(target_email_ids)}"}]
        else:
            return [{"status": "no_emails", "message": "No unread emails found"}]

    processed_ids = []

    for email in emails:
        try:
            # Strict validation before processing
            if not email or not email.id:
                print(f"[Email] ❌ Invalid email format - skipping")
                continue

            if not email.sender:
                print(f"[Email] ⚠️ Email {email.id} has no sender - skipping")
                continue

            if not email.subject:
                print(f"[Email] ⚠️ Email {email.id} has no subject - skipping")
                continue

            print(f"[Email] 🔍 Strict validation passed for: {email.subject} (ID: {email.id})")

            # Generate AI reply
            reply = generate_ai_reply(
                email=email,
                user_name=user_name,
                user_context=user_context,
                custom_instructions=custom_instructions,
            )

            # Create draft with strict validation
            draft = create_draft_reply(
                email=email,
                reply_body=reply,
                user_email=user_email,
            )

            results.append({
                "status": "draft_created",
                "email_id": email.id,
                "thread_id": email.thread_id,
                "sender": email.sender,
                "subject": email.subject,
                "date": email.date.isoformat(),
                "snippet": email.snippet[:100],
                "draft_subject": draft.subject,
                "draft_body_preview": reply[:150] + ("..." if len(reply) > 150 else ""),
            })
            processed_ids.append(email.id)

        except TokenExpiredError:
            raise
        except Exception as e:
            print(f"[Email] ❌ Failed to process email {email.id} after strict validation: {e}")
            results.append({
                "status": "error",
                "email_id": email.id,
                "subject": email.subject,
                "error": str(e),
            })

    # Mark as read if requested
    if mark_read_after and processed_ids:
        mark_as_read(processed_ids)

    return results
# ───────────────────────────────────────────────────────────────────────────────
# Tool Interface for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

def email_action(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """
    JARVIS tool entry point for email operations.

    Supported actions:
    - fetch: Fetch recent unread emails (returns summaries)
    - process: Fetch + generate AI replies + create drafts
    - draft: Create a draft reply for a specific email (requires email_id)
    - send: Send an email immediately (requires to, subject, body) or reply to an email (requires email_id)
    - reply: Reply to a specific email via AI-generated or custom body (requires email_id)
    - summarize: Fetch emails and generate AI summaries for each
    - auth: Force re-authentication
    - status: Check authentication status

    Parameters:
        action: "fetch" | "process" | "draft" | "send" | "reply" | "summarize" | "auth" | "status"
        max_results: Max emails to fetch (default: 10)
        days_back: Days back to search (default: 7)
        include_read: Include read emails in fetch results (default: false)
        query: Custom Gmail search query (e.g. "from:boss@example.com")
        email_id: Specific email ID for draft/send/reply action
        to: Recipient email address (for send action)
        subject: Email subject (for send action)
        body: Email body (for send action). If empty and replying via email_id, AI generates the reply.
        use_ai: Generate reply body via AI when replying (default: true)
        custom_instructions / instructions: Additional AI instructions
        mark_read: Whether to mark as read after processing (default: true)
    """
    action = parameters.get("action", "fetch").lower()
    max_results = int(parameters.get("max_results", DEFAULT_MAX_RESULTS))
    days_back = int(parameters.get("days_back", DEFAULT_DAYS_BACK))
    include_read = parameters.get("include_read", False)
    query = parameters.get("query", "").strip() or None
    # Accept both parameter names for compatibility
    custom_instructions = parameters.get("custom_instructions") or parameters.get("instructions", "")
    mark_read = parameters.get("mark_read", True)
    # Send/reply action parameters
    to_addr = parameters.get("to", "").strip()
    subject = parameters.get("subject", "").strip()
    body = parameters.get("body", "").strip()
    use_ai = parameters.get("use_ai", True)

    # Load user config
    user_name = load_api_keys().get("user_name", "")
    user_email = get_user_email()
    user_context = get_user_context()

    if player:
        player.write_log(f"[email] Action: {action}")

    try:
        if action == "auth":
            force_reauth()
            # Trigger new auth
            get_gmail_service()
            return "✅ Re-authentication complete. Gmail access granted."

        elif action == "status":
            creds = _load_token()
            if creds and creds.valid:
                expiry = creds.expiry.strftime("%Y-%m-%d %H:%M") if creds.expiry else "unknown"
                return f"✅ Gmail authenticated (token expires: {expiry})"
            elif creds:
                return "⚠️ Gmail token exists but expired — will auto-refresh on next use"
            else:
                return "❌ Not authenticated. Run with action='auth' to set up."

        elif action == "fetch":
            emails = fetch_unread_emails(
                max_results=max_results,
                days_back=days_back,
                include_read=include_read,
                query=query,
            )
            if not emails:
                label = "emails" if include_read else "unread emails"
                return f"📭 No {label} found."

            label = "emails" if include_read else "unread emails"
            lines = [f"📬 Found {len(emails)} {label}:\n"]
            for i, e in enumerate(emails, 1):
                sender_name = e.sender.split("<")[0].strip() if "<" in e.sender else e.sender
                unread_tag = " [UNREAD]" if e.is_unread else ""
                lines.append(f"{i}. **{sender_name}** — {e.subject}{unread_tag}")
                lines.append(f"   📅 {e.date.strftime('%Y-%m-%d %H:%M')}  |  📝 {e.snippet[:80]}...")
                lines.append(f"   ID: {e.id}")
                lines.append("")
            return "\n".join(lines)

        elif action == "process":
            results = process_unread_emails(
                max_results=max_results,
                days_back=days_back,
                include_read=include_read,
                query=query,
                user_name=user_name,
                user_context=user_context,
                user_email=user_email,
                custom_instructions=custom_instructions,
                mark_read_after=mark_read,
            )

            # Format results for user
            drafts = [r for r in results if r.get("status") == "draft_created"]
            errors = [r for r in results if r.get("status") == "error"]

            lines = [f"✅ Processed {len(results)} email(s) — {len(drafts)} draft(s) created"]
            if errors:
                lines.append(f"⚠️ {len(errors)} error(s)")

            for r in drafts:
                lines.append(f"\n📧 **{r['sender']}** — {r['subject']}")
                lines.append(f"   Draft: {r['draft_body_preview']}")

            for r in errors:
                lines.append(f"\n❌ {r['subject']}: {r['error']}")

            if player:
                player.write_log(f"[email] Created {len(drafts)} draft(s)")
            return "\n".join(lines)

        elif action == "draft":
            email_id = parameters.get("email_id")
            if not email_id:
                return "❌ Please provide email_id for draft action"

            # Fetch specific email
            service = get_gmail_service()
            msg = _execute_with_retry(
                service.users().messages().get(userId="me", id=email_id, format="full")
            )
            email = _parse_gmail_message(msg)

            # Generate reply
            reply = generate_ai_reply(
                email=email,
                user_name=user_name,
                user_context=user_context,
                custom_instructions=custom_instructions,
            )

            # Create draft
            draft = create_draft_reply(email=email, reply_body=reply, user_email=user_email)

            return (
                f"✅ Draft created for: {email.subject}\n"
                f"To: {draft.to}\n"
                f"Subject: {draft.subject}\n\n"
                f"Preview:\n{reply[:300]}..."
            )

        elif action == "send":
            email_id = parameters.get("email_id")

            # Mode 1: Reply to an existing email (requires email_id)
            if email_id:
                service = get_gmail_service()
                msg = _execute_with_retry(
                    service.users().messages().get(userId="me", id=email_id, format="full")
                )
                email = _parse_gmail_message(msg)

                # Generate reply body via AI if not provided
                if not body and use_ai:
                    reply = generate_ai_reply(
                        email=email,
                        user_name=user_name,
                        user_context=user_context,
                        custom_instructions=custom_instructions,
                    )
                else:
                    reply = body

                # Send the email as a reply in the same thread
                result = send_email(
                    to=email.sender,
                    subject=email.subject if email.subject.lower().startswith("re:") else f"Re: {email.subject}",
                    body=reply,
                    user_email=user_email,
                    thread_id=email.thread_id,
                    in_reply_to=email.id,
                    references=f"{email.references} {email.id}".strip() if email.references else email.id,
                )

                return (
                    f"✅ Email sent to {result['to']}\n"
                    f"Subject: {result['subject']}\n"
                    f"Message ID: {result['message_id']}\n\n"
                    f"Preview:\n{reply[:300]}..."
                )

            # Mode 2: Send a new email (requires to, subject, body)
            if not to_addr:
                return "❌ Please provide 'to' (recipient email address) for send action"
            if not subject:
                return "❌ Please provide 'subject' for send action"
            if not body:
                return "❌ Please provide 'body' for send action"

            result = send_email(
                to=to_addr,
                subject=subject,
                body=body,
                user_email=user_email,
            )

            return (
                f"✅ Email sent to {result['to']}\n"
                f"Subject: {result['subject']}\n"
                f"Message ID: {result['message_id']}"
            )

        elif action == "reply":
            email_id = parameters.get("email_id")
            if not email_id:
                return "❌ Please provide 'email_id' to reply to a specific email"

            # Fetch the email to reply to
            service = get_gmail_service()
            msg = _execute_with_retry(
                service.users().messages().get(userId="me", id=email_id, format="full")
            )
            email = _parse_gmail_message(msg)

            # Generate reply body via AI if not provided
            if not body and use_ai:
                reply = generate_ai_reply(
                    email=email,
                    user_name=user_name,
                    user_context=user_context,
                    custom_instructions=custom_instructions,
                )
            else:
                reply = body

            # Send the reply in the same thread
            result = send_email(
                to=email.sender,
                subject=email.subject if email.subject.lower().startswith("re:") else f"Re: {email.subject}",
                body=reply,
                user_email=user_email,
                thread_id=email.thread_id,
                in_reply_to=email.id,
                references=f"{email.references} {email.id}".strip() if email.references else email.id,
            )

            return (
                f"✅ Replied to {email.sender}\n"
                f"Subject: {result['subject']}\n"
                f"Message ID: {result['message_id']}\n\n"
                f"Preview:\n{reply[:300]}..."
            )

        elif action == "summarize":
            emails = fetch_unread_emails(
                max_results=max_results,
                days_back=days_back,
                include_read=include_read,
                query=query,
            )
            if not emails:
                label = "emails" if include_read else "unread emails"
                return f"📭 No {label} found to summarize."

            label = "emails" if include_read else "unread emails"
            lines = [f"📋 Summarizing {len(emails)} {label}:\n"]

            for i, email in enumerate(emails, 1):
                sender_name = email.sender.split("<")[0].strip() if "<" in email.sender else email.sender
                try:
                    summary = summarize_email(
                        email=email,
                        custom_instructions=custom_instructions,
                    )
                    unread_tag = " [UNREAD]" if email.is_unread else ""
                    lines.append(f"{i}. **{sender_name}** — {email.subject}{unread_tag}")
                    lines.append(f"   📅 {email.date.strftime('%Y-%m-%d %H:%M')}")
                    lines.append(f"   📝 Summary: {summary}")
                    lines.append("")
                except Exception as e:
                    lines.append(f"{i}. **{sender_name}** — {email.subject}")
                    lines.append(f"   ❌ Summary failed: {e}")
                    lines.append("")

            return "\n".join(lines)

        else:
            return f"❌ Unknown action: {action}. Use: fetch, process, draft, send, reply, summarize, auth, status"

    except TokenExpiredError:
        return ("❌ Authentication expired. Please run email action with action='auth' "
                "to re-authenticate with Google.")
    except AuthenticationError as e:
        return f"❌ Authentication error: {e}"
    except RateLimitError:
        return "❌ Rate limit exceeded. Please wait a moment and try again."
    except NetworkError as e:
        return f"❌ Network error: {e}"
    except EmailError as e:
        return f"❌ Email error: {e}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Unexpected error: {e}"
# ───────────────────────────────────────────────────────────────────────────────
# Tool Declaration for JARVIS
# ───────────────────────────────────────────────────────────────────────────────

EMAIL_TOOL_DECLARATION = {
    "name": "email",
    "description": (
        "Manages Gmail via API: fetch emails, generate AI replies, create drafts, "
        "send emails, and summarize emails. Actions: fetch (list emails), process (fetch + AI drafts), "
        "draft (create draft reply), send (send email immediately), "
        "reply (reply to a specific email via AI), summarize (generate AI summaries of emails), "
        "auth (re-authenticate), status (check auth)."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "fetch | process | draft | send | reply | summarize | auth | status",
                "enum": ["fetch", "process", "draft", "send", "reply", "summarize", "auth", "status"],
            },
            "max_results": {
                "type": "INTEGER",
                "description": "Maximum emails to fetch (default: 10, max: 50)",
            },
            "days_back": {
                "type": "INTEGER",
                "description": "How many days back to search (default: 7)",
            },
            "include_read": {
                "type": "BOOLEAN",
                "description": "Include read emails in fetch results (default: false, only unread)",
            },
            "query": {
                "type": "STRING",
                "description": "Custom Gmail search query (e.g. 'from:boss@example.com')",
            },
            "email_id": {
                "type": "STRING",
                "description": "Specific email ID for draft, send, or reply action",
            },
            "to": {
                "type": "STRING",
                "description": "Recipient email address (for send action)",
            },
            "subject": {
                "type": "STRING",
                "description": "Email subject line (for send action)",
            },
            "body": {
                "type": "STRING",
                "description": "Email body (for send action). If empty and replying via email_id, AI generates the reply.",
            },
            "use_ai": {
                "type": "BOOLEAN",
                "description": "Generate reply body via AI when replying to an email (default: true)",
            },
            "custom_instructions": {
                "type": "STRING",
                "description": "Additional instructions for AI reply generation",
            },
            "mark_read": {
                "type": "BOOLEAN",
                "description": "Mark emails as read after processing (default: true)",
            },
        },
        "required": ["action"],
    },
}

# ───────────────────────────────────────────────────────────────────────────────
# Setup Helper
# ───────────────────────────────────────────────────────────────────────────────

def setup_gmail_credentials(credentials_json: str) -> bool:
    """
    Save Gmail OAuth credentials from JSON string.

    Args:
        credentials_json: Full OAuth client credentials JSON from Google Cloud Console

    Returns:
        True if saved successfully
    """
    try:
        data = json.loads(credentials_json)
        # Validate structure
        if "installed" not in data and "web" not in data:
            raise ValueError("Invalid credentials format — must be OAuth client ID for Desktop or Web app")

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CREDENTIALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[Email] ✅ Credentials saved to {CREDENTIALS_FILE}")
        return True
    except json.JSONDecodeError as e:
        print(f"[Email] ❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"[Email] ❌ Failed to save credentials: {e}")
        return False
def get_setup_instructions() -> str:
    """Return setup instructions for Gmail API."""
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GMAIL API SETUP FOR JARVIS                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
1. Go to Google Cloud Console:                                             ║
   https://console.cloud.google.com/                                       ║
                                                                              ║
2. Create a new project or select existing one                            ║
                                                                              ║
3. Enable Gmail API:                                                       ║
   APIs & Services → Library → Search "Gmail API" → Enable                ║
                                                                              ║
4. Create OAuth 2.0 Credentials:                                           ║
   APIs & Services → Credentials → Create Credentials → OAuth Client ID   ║
   - Application type: Desktop Application                                 ║
   - Name: JARVIS Assistant                                                ║
   - Authorized redirect URIs: http://localhost:8080/                     ║
                                                                              ║
5. Download the credentials JSON file                                      ║
                                                                              ║
6. Save as: {CREDENTIALS_FILE}                                           ║
                                                                              ║
7. Add your user info to config/api_keys.json:                             ║
   {{                                                                      ║
     "user_name": "Your Name",                                             ║
     "user_email": "your.email@gmail.com",                                 ║
     "user_context": "Your role/context for AI replies"                    ║
   }}                                                                      ║
                                                                              ║
8. Run JARVIS and use: "JARVIS, check my email" or "process my emails"    ║
                                                                              ║
   ⚠️  First run will open browser for Google OAuth consent.                 ║
      Tokens are stored securely in: {TOKEN_FILE}                           ║
                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# ───────────────────────────────────────────────────────────────────────────────
# Module Test
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Email Module")
    parser.add_argument("action", nargs="?", default="status",
                        choices=["fetch", "process", "auth", "status", "setup"])
    parser.add_argument("--max", type=int, default=10, help="Max emails")
    parser.add_argument("--days", type=int, default=7, help="Days back")
    parser.add_argument("--instructions", type=str, default="", help="Custom AI instructions")
    args = parser.parse_args()

    print(get_setup_instructions())

    if args.action == "setup":
        print("Paste your OAuth credentials JSON (Ctrl+D to end):")
        import sys
        creds_json = sys.stdin.read()
        if setup_gmail_credentials(creds_json):
            print("✅ Credentials saved. Run with 'auth' to authenticate.")
        sys.exit(0)

    params = {
        "action": args.action,
        "max_results": args.max,
        "days_back": args.days,
        "custom_instructions": args.instructions,
    }

    result = email_action(params)
    print(result)