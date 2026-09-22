# ⚙️ MARK XLIX
### The Ultimate Cross-Platform Personal AI Assistant — By FatihMakes

> 📺 **[Watch the full setup video on YouTube](https://youtu.be/CiGdcIlnXb8))**

A real-time voice AI that can hear, see, understand, and control your computer — on any OS. Supports Windows, macOS, and Linux. Built on the Gemini Live API for native audio streaming, delivering zero subscriptions and total digital autonomy.

---

## ✨ Overview

MARK XLIX deepens the personal assistant foundation. Rather than adding more tools, this build focused on making the assistant truly *yours*: it starts with your computer, learns your name, and pays attention to what you're doing. The goal before the plugin era begins is a core that feels alive — not just reactive.

---

## 🚀 Capabilities

### Core Features
| Feature | Description |
|---|---|
| 🎙️ Real-time Voice | Ultra-low latency conversation in any language via Gemini Live API |
| 🖥️ System Control | Launch apps, adjust volume/brightness, WiFi, shortcuts, power — all by voice |
| 🧩 Autonomous Tasks | High-level planning for complex multi-step goals via agent mode |
| 👁️ Visual Awareness | Real-time screen capture and webcam vision piped into your main Gemini session |
| 🧠 Persistent Memory | Deeply remembers projects, preferences, and personal context across sessions |
| 👤 Face Recognition | Remembers people by face — enroll with "This is Alice", recognize later across sessions |
| 🎥 Visual Memory | Stores images, videos, and keyframes with detected people, objects, OCR text, and scene descriptions |
| 🔗 Episodic Memory | Links people, media, and events into a searchable timeline of your life |
| ⌨️ Hybrid Input | Seamlessly switch between keyboard typing and voice commands |
| 🌅 Morning Briefing | On first boot: greets you, reads the time, fetches live news headlines, and checks weather |
| 🔔 Proactive Check-ins | After 15 minutes of silence, checks context and offers something genuinely useful |
| 📊 Hardware Monitoring | Continuous CPU, RAM, GPU and temperature telemetry with localized voice alerts |
| 🌤️ Weather Report | Live weather data for your city, personalized from memory |
| 🗺️ Dynamic Content Panel | Scrollable display layer beneath the HUD that renders web results, news, and search data |
| 🔍 Multi-Mode Web Search | `news` / `research` / `price` / `compare` / `search` — Gemini Grounded first, DDG fallback |
| ⏰ Smart Reminders | OS-native scheduled notifications (Windows Task Scheduler / macOS LaunchAgent / Linux systemd) |
| ✈️ Flight Finder | Live flight price and availability lookup |
| 🎮 Game Updater | Checks and triggers game updates on Steam and Epic Games on demand |
| 📂 File Processor | Read, summarize, and answer questions about local files |
| 💻 Code Helper | Inline code review, debugging, and generation |
| 🌐 Browser Control | Open URLs, navigate tabs, and interact with the browser by voice |
| 📨 Send Message | Compose and send messages through WhatsApp, Telegram, and more |
| 📧 Email | Gmail API integration: fetch unread emails, AI-generated draft replies, human review |
| 📅 Calendar | Google Calendar API integration: list events, create/update/cancel meetings, free/busy lookup, optional Meet link |
| 👥 Teams Calendar | Microsoft Graph API integration: list events, create/update/cancel meetings, free/busy lookup, optional Teams meeting link |
| 🗓️ Meeting Scheduler | Unified scheduling across Google/Teams calendars: auto provider selection, auto free-slot search, merged "what's on my calendar" view |
| 🎬 YouTube Control | Search, play, and control YouTube playback by voice |
| 🖱️ Desktop Control | Taskbar, window management, and desktop-level operations |
| 🧑‍💻 Silent Language Memory | Detects spoken language on first use and saves it — all future sessions adapt automatically |
| 📱 Remote Dashboard | Control the assistant from your phone via QR code pairing |

---

## 🆕 What's New in XLIX

### ⚡ Auto-Start on Boot
The assistant now registers itself with the operating system's startup system. One click in the UI toggles it on or off. On Windows, it writes to the registry using `pythonw.exe` so no console window ever appears. On macOS it installs a LaunchAgent plist; on Linux a `.desktop` autostart entry. The button reflects the current state every time the app launches.

### 🎨 Assistant Customization
The assistant is no longer locked to the name "JARVIS". Click `⚙ CUSTOMISE ASSISTANT` in the right panel to change:
- **Assistant name** — displayed everywhere in the UI (title bar, header, HUD, log, footer) and injected into the Gemini system prompt so the AI knows its own name
- **Your name** — how the assistant addresses you. Leave blank for the default language-aware addressing (`sir` / `efendim`), or set your actual name for a more personal feel

Changes take effect immediately without restarting.

### 📋 Clipboard Intelligence
Copy any text of 10 or more characters and a floating panel appears at the bottom of the window. Four quick actions — **TRANSLATE**, **SUMMARISE**, **EXPLAIN**, **FIX** — send the copied content directly to the assistant with one click. The panel auto-dismisses after 8 seconds. This turns the clipboard into a silent command channel for anything on your screen.

### ☀ Morning Brief Toggle + Speed Optimization
The morning briefing can now be turned on or off with one click from the settings drawer (`⚙` → `☀ MORNING BRIEF: ON/OFF`). Users who don't want a startup briefing can disable it permanently; the setting survives restarts. The briefing itself was also re-engineered: news is now pre-fetched in a background thread the moment the session starts, running in parallel while the greeting plays. By the time the greeting finishes, the results are already ready — no extra Gemini tool-call round-trip needed. Briefing delivery is noticeably faster as a result.

---

## 🗺️ Mark Roadmap

| Mark | Focus |
|---|---|
| **XLVIII** | Instant interrupt · parallel news · two-phase briefing · exponential backoff · vision cooldown |
| **XLIX** | Auto-start · clipboard intelligence · assistant customization |
| **L** | Wake word · proactive system 2.0 · session memory / daily continuity |
| **LI+** | Plugin system · email · quiz mode · calorie counter · and more |

---

## ⚡ Quick Start

```bash
git clone https://github.com/FatihMakes/Mark-XLIX.git
cd Mark-XLIX
pip install -r requirements.txt
python main.py
```

> ⚠️ **Installation Note:** Some OS-specific dependencies are not bundled in `requirements.txt` to keep the repo lightweight. If you hit a `ModuleNotFoundError`, install the missing package with `pip install <module_name>`.

---

## 📋 Requirements

| Requirement | Details |
| --- | --- |
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | 3.11 or 3.12 |
| **Microphone** | Required for voice interaction |
| **API Key** | Free Gemini API key (`config/api_keys.json`) |

---

## 🗂️ Project Structure

```
Mark XLIX/
├── main.py                  # Core loop — Gemini Live session, audio I/O, tool dispatch
├── ui.py                    # PyQt6 HUD — waveform, log panel, interrupt button, camera feed
├── setup.py                 # First-run configuration wizard
├── actions/
│   ├── web_search.py        # Gemini + DDG parallel search (news, research, price, compare)
│   ├── screen_processor.py  # Screen capture & webcam vision via Gemini Live
│   ├── reminder.py          # OS-native scheduled notifications
│   ├── system_monitor.py    # CPU / RAM / GPU / temperature telemetry
│   ├── computer_settings.py # Volume, brightness, WiFi, power
│   ├── computer_control.py  # Keyboard shortcuts, mouse, window management
│   ├── open_app.py          # Application launcher
│   ├── browser_control.py   # Web browser control
│   ├── file_controller.py   # File system operations
│   ├── file_processor.py    # Document reading and summarization
│   ├── send_message.py      # Messaging integration
│   ├── weather_report.py    # Live weather data
│   ├── flight_finder.py     # Flight search
│   ├── youtube_video.py     # YouTube playback control
│   ├── game_updater.py      # Game update management (Steam / Epic)
│   ├── code_helper.py       # Code review and generation
│   ├── dev_agent.py         # Developer task agent
│   ├── desktop.py           # Desktop and taskbar control
│   ├── proactive.py         # Proactive silence-break suggestions
│   ├── email.py             # Gmail API: fetch, AI replies, draft creation
│   ├── calendar_google.py   # Google Calendar API: list/create/update/delete events, free/busy
│   ├── calendar_teams.py    # Microsoft Graph API: list/create/update/delete events, free/busy, Teams links
│   └── meeting_scheduler.py # Unified scheduler: provider auto-resolve, free-slot search, merged calendar view
├── memory/                  # Persistent memory system (SQLite + JSON)
│   ├── db.py                # SQLite schema, CRUD, migrations
│   ├── memory_manager.py    # Textual memory API (load/save/update/forget)
│   ├── vector_store.py      # Embedding storage (SQLite BLOB, FAISS-ready)
│   ├── face_memory.py       # Face enrollment, recognition, history
│   ├── media_memory.py      # Image/video/keyframe storage and queries
│   ├── event_memory.py      # Episodic memory linking people, media, events
│   └── config_manager.py    # Configuration management
├── recognition/             # Modular CV backends
│   └── face_backend.py      # InsightFace face detection + embedding
├── tests/
│   └── test_visual_memory.py # Integration tests for visual memory
├── core/
│   └── prompt.txt           # Assistant personality and tool-routing rules
└── config/
    └── api_keys.json        # API key, OS setting, assistant name, user name
```
---

## 👤 Face Recognition & Visual Memory

### Overview

MARK XLIX can remember people by their face and store rich visual memories. When someone is introduced, their face is detected, a 512-dimensional embedding is extracted using InsightFace, and stored in the database. Later, whenever the camera captures a face, the system identifies who it is with a confidence score.

### How It Works

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│                    Application Layer                 │
│  (main.py — Gemini Live, voice, tool dispatch)       │
├─────────────────────────────────────────────────────┤
│                   Memory Services                    │
│  FaceMemory  — enrollment, recognition, history      │
│  MediaMemory — image/video/keyframe storage          │
│  EventMemory — links people, media, events           │
│  VectorStore — embedding similarity search           │
├─────────────────────────────────────────────────────┤
│                 Recognition Layer                    │
│  FaceBackend — InsightFace (detection + embedding)   │
├─────────────────────────────────────────────────────┤
│                   Storage Layer                      │
│  SQLite (metadata) + float32 BLOBs (embeddings)      │
│  File system (images, videos, keyframes)             │
└─────────────────────────────────────────────────────┘
```

**Database Schema:**

| Table | Purpose |
|---|---|
| `people` | Person records (name, first/last seen, embedding dim) |
| `person_embeddings` | float32 BLOB embeddings (multiple per person) |
| `media` | Images, videos, clips with metadata |
| `keyframes` | Extracted frames from videos with timestamps |
| `detected_objects` | Object detections (label, confidence, bbox) |
| `detected_people` | Person detections linked to people table |
| `ocr_text` | OCR text extracted from media |
| `scene_descriptions` | LLM-generated scene descriptions |
| `events` | Episodic memory events (introductions, meetings, sightings) |
| `event_media` | Junction: events ↔ media |
| `event_people` | Junction: events ↔ people |

### Voice Commands

| Command | What It Does |
|---|---|
| **"This is Alice"** | Captures a camera frame, detects the face, creates a person record, stores the embedding, and creates an introduction event |
| **"Who is this?"** / **"Who is in front of the camera?"** | Captures a camera frame, detects faces, and identifies them against known people |
| **"When did you last see Alice?"** | Returns the most recent event involving that person |
| **"When did you first meet Bob?"** | Returns the first event involving that person |
| **"Find videos containing both Alice and Charlie"** | Finds media where both people were detected |

### Key Design Decisions

1. **InsightFace** — Uses the `buffalo_l` model with RetinaFace detection and ArcFace R100 recognition (512-dim embeddings, CPU-compatible)
2. **Multiple embeddings per person** — Each enrollment adds a new embedding, improving recognition accuracy over time
3. **Auto-enrollment** — When recognition confidence exceeds 0.85, the embedding is automatically saved to improve future recognition
4. **Normalized schema** — Every entity is its own table with foreign keys; no JSON blobs for structured data
5. **Vector store abstraction** — The `VectorStore` ABC allows swapping SQLite for FAISS or pgvector without changing callers
6. **Media decomposition** — Videos are decomposed into keyframes, each with its own detections and metadata
7. **Backward compatible** — The existing textual memory system is unchanged; new tables are additive

### Installation

InsightFace and ONNX Runtime are included in `requirements.txt`. On first use, the model (~400 MB) is downloaded automatically.

```bash
pip install insightface onnxruntime
```

### Storage Location

Visual memories are stored under `memory/storage/media/`:

```
memory/storage/media/
├── images/
│   └── 2026-07-24/
│       ├── <sha256[:16]>.jpg
├── videos/
│   └── 2026-07-24/
│       ├── <sha256[:16]>.mp4
└── keyframes/
    └── <media_id>/
        ├── <media_time>.jpg
```

---

## 📧 Gmail API Setup

To enable the Email module, you need to configure Gmail API access:

1. **Go to Google Cloud Console**: https://console.cloud.google.com/

2. **Create or select a project**

3. **Enable Gmail API**:
   - APIs & Services → Library → Search "Gmail API" → Enable

4. **Create OAuth 2.0 Credentials**:
   - APIs & Services → Credentials → Create Credentials → OAuth Client ID
   - Application type: **Desktop Application**
   - Name: "JARVIS Assistant"
   - Authorized redirect URIs: `http://localhost:8080/`

5. **Download the credentials JSON** and save as:
   ```
   config/gmail_credentials.json
   ```

6. **Add user info to config/api_keys.json**:
   ```json
   {
     "gemini_api_key": "your-gemini-key",
     "os_system": "windows",
     "user_name": "Your Name",
     "user_email": "your.email@gmail.com",
     "user_context": "Brief description of your role/context for AI replies"
   }
   ```

7. **First run**: Say "JARVIS, check my email" or use the `email` tool with `action: auth`. A browser window will open for Google OAuth consent. Tokens are stored securely in `config/gmail_token.json`.

### Email Commands
- **"Check my email"** — Fetch recent unread emails
- **"Process my emails"** — Fetch unread, generate AI replies, create drafts
- **"Email status"** — Check authentication status

> ⚠️ **Security**: Emails are fetched via Gmail API (no browser automation). Drafts are created for your review — nothing is sent automatically.
```

---

## 📅 Google Calendar Setup

The Calendar module reuses the same OAuth client as Gmail — if you've already done the Gmail setup above, you only need step 2 below.

1. **Go to Google Cloud Console**: https://console.cloud.google.com/ (same project as Gmail)

2. **Enable the Calendar API**:
   - APIs & Services → Library → Search "Google Calendar API" → Enable

3. **Reuse (or create) OAuth credentials**: if `config/gmail_credentials.json` already exists from the Gmail setup, no action needed — Calendar authenticates separately (its own token file) but shares the same OAuth client.

4. **First run**: Say "JARVIS, what's on my calendar" or use the `calendar_google` tool with `action: auth`. A browser window will open for Google OAuth consent (separate consent screen from Gmail, since the scope is different). Tokens are stored in `config/gcal_token.json`.

### Calendar Commands
- **"What's on my calendar today"** — List upcoming events
- **"Schedule a meeting with [name] tomorrow at 3pm"** — Create an event
- **"Add a Google Meet link to that"** — Create an event with `add_meet_link: true`
- **"Cancel my 3pm meeting"** — Delete an event
- **"Calendar status"** — Check authentication status

> ⚠️ Attendees are invited via calendar invite email (`sendUpdates: all`) when the event has attendees — nothing is sent silently.

---

## 👥 Microsoft Teams Calendar Setup

The Teams Calendar module talks to Microsoft 365 / Outlook via the Microsoft Graph API. Unlike Gmail/Google Calendar, it needs its own Azure app registration.

1. **Go to the Azure Portal**: https://portal.azure.com/ → Entra ID → App registrations → New registration
   - Supported account types: **Accounts in this organizational directory only** (single tenant) — a tenant admin must grant consent in step 3.
   - Redirect URI: leave blank here.

2. **Enable public client flows**:
   - App registration → Authentication → Add a platform → **Mobile and desktop applications** → check `http://localhost`
   - Set **Allow public client flows** to **Yes**

3. **Add API permissions** (App registration → API permissions → Add a permission → Microsoft Graph → Delegated permissions):
   - `Calendars.ReadWrite`
   - `User.Read`
   - `OnlineMeetings.ReadWrite`
   - Click **Grant admin consent for [tenant]** (requires a tenant admin)

4. **Copy your IDs** from the app's Overview page: **Application (client) ID** and **Directory (tenant) ID**

5. **Add to config/api_keys.json**:
   ```json
   {
     "ms_client_id": "your-application-client-id",
     "ms_tenant_id": "your-directory-tenant-id"
   }
   ```

6. **First run**: Say "JARVIS, Teams calendar status" or use the `calendar_teams` tool with `action: auth`. A browser window opens for Microsoft sign-in/consent. Tokens are cached in `config/msgraph_token_cache.json`.

### Teams Calendar Commands
- **"What's on my Teams calendar today"** — List upcoming events
- **"Schedule a Teams meeting with [name] tomorrow at 2pm"** — Create an event with a Teams join link (`is_teams_meeting: true`)
- **"Cancel my 2pm Teams meeting"** — Delete an event
- **"Teams calendar status"** — Check authentication status

> ⚠️ **Admin consent**: if the tenant admin hasn't granted consent for the requested permissions, sign-in will fail with an `AADSTS65001` error. Have your Azure admin grant consent from the app registration's API permissions page.

---

## 🗓️ Meeting Scheduler (Unified)

`meeting_scheduler` sits on top of Google Calendar and Teams Calendar so you don't have to say which one — it picks a provider automatically and, when you don't give an exact time, finds the next open slot itself.

- **Provider resolution**: explicit request > `calendar_provider_default` in `config/api_keys.json` (`"google"` or `"teams"`) > whichever single provider is configured > Google as a last-resort tie-break when both are set up.
- **Auto slot-finding**: if you don't name a time, it searches business hours (09:00–18:00 UTC) up to 5 days ahead on the resolved provider's calendar for the first gap that fits the requested duration.
- **Merged listing**: "what's on my calendar" pulls from every configured provider and sorts them together, each tagged 🟦 (Google) or 🟪 (Teams).
- **Composite IDs**: list results carry an id like `google:abc123` or `teams:xyz789` — pass that straight back to cancel or reschedule that specific meeting.

### Meeting Scheduler Commands
- **"What's on my calendar today"** — merged view across both providers
- **"Schedule a 30-minute meeting with [name] tomorrow"** — auto-finds a free slot, no time needed
- **"Set up a call with [name] at 2pm on Friday"** — exact time, provider auto-resolved
- **"Cancel my meeting with [name]"** — uses the composite id from a prior list
- **"Move my 2pm to 4pm"** — reschedule

> ⚠️ Auto slot-finding checks the organizer's own calendar (and, for Teams, any attendees in the same tenant via `getSchedule`). It does not see personal Google calendars of external attendees — those recipients get invited via email and can propose a new time themselves.

---

## ⚠️ License

Personal and non-commercial use only.
Licensed under **[Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**.

---

## 👤 Connect with the Creator

Engineered by a developer building a real-world JARVIS-style assistant.
⭐ **Star the repository to support the journey to Mark 100.**

| Platform | Link |
| --- | --- |
| YouTube | [@FatihMakes](https://www.youtube.com/@FatihMakes) |
| Instagram | [@fatihmakes](https://www.instagram.com/fatihmakes) |
