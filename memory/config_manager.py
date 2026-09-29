import json
import sys
from pathlib import Path

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR    = get_base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"

def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def config_exists() -> bool:
    return CONFIG_FILE.exists()

def save_api_keys(gemini_api_key: str) -> None:
    ensure_config_dir()

    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data["gemini_api_key"] = gemini_api_key.strip()

    CONFIG_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

def load_api_keys() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to load api_keys.json: {e}")
        return {}

def get_gemini_key() -> str | None:
    return load_api_keys().get("gemini_api_key")

def is_configured() -> bool:
    key = get_gemini_key()
    return bool(key and len(key) > 15)


def get_assistant_name() -> str:
    """Return the configured assistant name, or 'JARVIS' if not set."""
    return load_api_keys().get("assistant_name", "JARVIS") or "JARVIS"


def get_user_name() -> str:
    """Return the configured user name for addressing."""
    return load_api_keys().get("user_name", "")


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    """Persist assistant name and user name to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["assistant_name"] = assistant_name.strip() or "JARVIS"
    data["user_name"] = user_name.strip()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_brief_enabled() -> bool:
    return load_api_keys().get("morning_brief_enabled", True)


def save_brief_enabled(enabled: bool) -> None:
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["morning_brief_enabled"] = enabled
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_vad_silence_timeout_ms() -> int:
    """Return the VAD silence timeout in milliseconds.
    Default: 60000ms (60 seconds) - allows long JARVIS responses without cutoff.
    """
    return load_api_keys().get("vad_silence_timeout_ms", 60000)


def save_vad_silence_timeout_ms(timeout_ms: int) -> None:
    """Persist the VAD silence timeout to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["vad_silence_timeout_ms"] = int(timeout_ms)
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")

def get_user_email() -> str:
    """Return the configured user email for Gmail API."""
    return load_api_keys().get("user_email", "")


def save_user_email(email: str) -> None:
    """Save user email to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["user_email"] = email.strip().lower()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_user_context() -> str:
    """Return the configured user context for AI email replies."""
    return load_api_keys().get("user_context", "")


def save_user_context(context: str) -> None:
    """Save user context to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["user_context"] = context.strip()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


# ── Gemini Live voice configuration ────────────────────────────────────────────

# Prebuilt voices supported by gemini-2.5/2.0-flash-live. Full set:
LIVE_VOICES = (
    # Deep / authoritative
    "Charon", "Rasalgethi", "Alnilam", "Pulcherrima",
    # Neutral / warm / bright
    "Puck", "Kore", "Zephyr", "Aoede", "Leda",
    # Casual / character
    "Fenrir", "Callirrhoe", "Umbriel", "Erinome", "Achird",
    "Sadaltager", "Vindemiatrix", "Autonoe", "Orus", "Laomedeia",
    "Sadachbia", "Achernar", "Gacrux", "Zubenelgenubi",
    "Algieba", "Algenib", "Enceladus", "Iapetus", "Despina",
    "Schedar", "Sulafat",
)


def normalize_voice(name: str) -> str:
    """Return a valid voice name (case-insensitive match) or 'Charon' fallback."""
    if not name:
        return "Charon"
    for v in LIVE_VOICES:
        if v.lower() == name.strip().lower():
            return v
    return "Charon"


def get_ms_client_id() -> str:
    """Return the Azure AD app registration client (application) ID for Microsoft Graph."""
    return load_api_keys().get("ms_client_id", "")


def get_ms_tenant_id() -> str:
    """Return the Azure AD tenant ID. Defaults to 'organizations' (any work/school tenant)."""
    return load_api_keys().get("ms_tenant_id", "") or "organizations"


def save_ms_app_config(client_id: str, tenant_id: str = "") -> None:
    """Persist Azure AD app registration details for Microsoft Graph (Teams Calendar)."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["ms_client_id"] = client_id.strip()
    data["ms_tenant_id"] = tenant_id.strip()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_calendar_provider_default() -> str:
    """Return the preferred calendar provider ('google' | 'teams') for meeting_scheduler
    when the user doesn't specify one. Empty string means auto-detect."""
    return load_api_keys().get("calendar_provider_default", "")


def save_calendar_provider_default(provider: str) -> None:
    """Persist the preferred default calendar provider."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["calendar_provider_default"] = provider.strip().lower()
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def get_voice() -> str:
    """Return the configured Gemini Live voice name (default 'Charon')."""
    return load_api_keys().get("live_voice", "Charon") or "Charon"


def save_voice(voice: str) -> None:
    """Persist the Gemini Live voice name to config."""
    ensure_config_dir()
    data: dict = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["live_voice"] = voice.strip() or "Charon"
    CONFIG_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")