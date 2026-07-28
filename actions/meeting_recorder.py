"""
Screen Recorder & Meeting Notes for MARK XLIX (JARVIS).

- ScreenRecorder: lightweight screen-capture video recorder (mss + cv2.VideoWriter).
  Records at a low, CPU-friendly frame rate (default 4 fps) — good enough to review
  what was on screen, without the overhead of a full 30fps capture.

- MeetingRecorder: combines a ScreenRecorder with raw mic audio capture (fed in
  from main.py's existing mic callback) into one timestamped session folder.
  On stop(), call summarize() to transcribe the audio and generate a structured
  meeting summary via Gemini.
"""
from __future__ import annotations

import json
import sys
import time
import wave
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    _MSS = True
except ImportError:
    _MSS = False

# Must match main.py's SEND_SAMPLE_RATE (mic is captured at 16kHz mono int16
# for the Gemini Live API, and we tap that same raw stream for recording).
SEND_SAMPLE_RATE = 16000

# Model for transcription + summarization. gemini-2.5-flash is deprecated/404ing
# as of mid-2026 — gemini-3.5-flash is the current GA multimodal model that
# accepts audio input directly.
SUMMARY_MODEL = "gemini-3.5-flash"


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def _get_api_key() -> str:
    cfg_path = _base_dir() / "config" / "api_keys.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        key = json.load(f).get("gemini_api_key", "")
    if not key:
        raise RuntimeError("gemini_api_key not found in config/api_keys.json")
    return key


class ScreenRecorder:
    """Records the primary monitor to an .mp4 file at a low frame rate."""

    def __init__(self, fps: int = 4):
        self.fps = fps
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self.output_path: Optional[Path] = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self, output_path: Path) -> None:
        if not _MSS or not _CV2:
            raise RuntimeError(
                "Screen recording needs 'mss' and 'opencv-python'. "
                "Run: pip install mss opencv-python"
            )
        if self._active:
            return
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._active = True
        self._thread.start()

    def stop(self) -> Optional[Path]:
        if not self._active:
            return None
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._active = False
        if self._error:
            raise RuntimeError(self._error)
        return self.output_path

    def _run(self):
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                target = monitors[1] if len(monitors) > 1 else monitors[0]
                width, height = target["width"], target["height"]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(self.output_path), fourcc, self.fps, (width, height)
                )
                if not writer.isOpened():
                    self._error = "Could not open video writer (codec/output path issue)."
                    return

                interval = 1.0 / self.fps
                try:
                    while not self._stop_event.is_set():
                        t0 = time.time()
                        shot = sct.grab(target)
                        frame = np.array(shot)[:, :, :3]  # BGRA -> BGR
                        writer.write(frame)
                        elapsed = time.time() - t0
                        time.sleep(max(0.0, interval - elapsed))
                finally:
                    writer.release()
        except Exception as e:
            self._error = str(e)


class MeetingRecorder:
    """
    Combines screen recording + raw mic audio capture into one session folder.

    Call feed_audio() continuously with raw int16 PCM mono 16kHz bytes — main.py's
    mic callback already produces exactly this format for the Gemini Live API, so
    it's just tapped and duplicated into the wav writer here.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self._root = base_dir or (_base_dir() / "meeting_notes")
        self._root.mkdir(exist_ok=True)
        self._screen = ScreenRecorder(fps=4)
        self._wav: Optional[wave.Wave_write] = None
        self._lock = threading.Lock()
        self._active = False
        self.session_dir: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> Path:
        if self._active:
            return self.session_dir  # type: ignore[return-value]

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self._root / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)

        wav_path = self.session_dir / "audio.wav"
        self._wav = wave.open(str(wav_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)  # int16
        self._wav.setframerate(SEND_SAMPLE_RATE)

        try:
            self._screen.start(self.session_dir / "screen.mp4")
        except Exception as e:
            # Still proceed with audio-only if screen capture isn't available.
            print(f"[MeetingRecorder] Screen recording unavailable: {e}")

        self._active = True
        return self.session_dir

    def feed_audio(self, pcm_bytes: bytes) -> None:
        if not self._active or self._wav is None:
            return
        with self._lock:
            try:
                self._wav.writeframes(pcm_bytes)
            except Exception:
                pass

    def stop(self) -> Optional[Path]:
        if not self._active:
            return None
        self._active = False
        try:
            self._screen.stop()
        except Exception as e:
            print(f"[MeetingRecorder] Screen recorder stop error: {e}")
        with self._lock:
            if self._wav:
                self._wav.close()
                self._wav = None
        return self.session_dir

    def summarize(self, session_dir: Path) -> str:
        """
        Upload the recorded audio to Gemini, transcribe it, then generate a
        structured markdown summary. Saves transcript.txt and summary.md into
        the session folder and returns the summary text.
        """
        from google import genai

        audio_path = session_dir / "audio.wav"
        if not audio_path.exists() or audio_path.stat().st_size < 2000:
            return "The recording was too short to summarize."

        client = genai.Client(api_key=_get_api_key())
        uploaded = client.files.upload(file=str(audio_path))

        transcript_resp = client.models.generate_content(
            model=SUMMARY_MODEL,
            contents=[
                uploaded,
                "Transcribe this recording verbatim. Label distinguishable speakers "
                "as Speaker 1, Speaker 2, etc.",
            ],
        )
        transcript = transcript_resp.text or ""
        (session_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

        summary_resp = client.models.generate_content(
            model=SUMMARY_MODEL,
            contents=[
                f"Here is a meeting transcript:\n\n{transcript}\n\n"
                "Write a concise meeting summary in markdown with these exact "
                "sections: '## Key Points', '## Decisions', '## Action Items' "
                "(include an owner name next to each item if one was mentioned). "
                "Only include what's actually supported by the transcript — do "
                "not invent details."
            ],
        )
        summary = summary_resp.text or "Could not generate a summary."
        (session_dir / "summary.md").write_text(summary, encoding="utf-8")

        return summary