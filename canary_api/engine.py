"""NVIDIA Canary ASR model engine with lazy loading and thread-safety."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo.collections.asr.models import ASRModel

LOGGER = logging.getLogger(__name__)


class CanaryEngine:
    """Thread-safe singleton for NVIDIA Canary model."""

    _instance: Optional[CanaryEngine] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize engine (use .instance() instead)."""
        self._model: Optional[ASRModel] = None
        self._model_lock = threading.Lock()
        self._model_ready = threading.Event()

    @classmethod
    def instance(cls) -> CanaryEngine:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_ready.is_set()

    def get_model(self) -> ASRModel:
        """Load and return the model (thread-safe, lazy)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    LOGGER.info("Loading NVIDIA Canary model 'nvidia/canary-1b-v2'...")
                    model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
                    model.to("cpu")
                    model.eval()
                    self._model = model
                    self._model_ready.set()
                    LOGGER.info("Model loaded successfully")
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        source_lang: str,
        target_lang: str,
        timestamps: bool,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to WAV audio file (16kHz mono)
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)
            timestamps: Whether to return timestamps

        Returns:
            Dictionary with 'text' and optional 'segments'
        """
        model = self.get_model()

        # Build transcription parameters
        kwargs: Dict[str, Any] = {
            "source_lang": source_lang,
            "target_lang": target_lang,
        }

        if timestamps:
            kwargs["return_hypotheses"] = True

        # Call model
        LOGGER.info(
            "Transcribing audio: source=%s, target=%s, timestamps=%s",
            source_lang,
            target_lang,
            timestamps,
        )

        try:
            result = model.transcribe([str(audio_path)], **kwargs)
        except TypeError as exc:
            # Fallback: model may not support all parameters
            LOGGER.warning("Model rejected parameters, retrying without them: %s", exc)
            result = model.transcribe([str(audio_path)])
            timestamps = False  # Disable timestamps on fallback

        # Parse result
        text, segments = _parse_result(result, timestamps)
        return {"text": text, "segments": segments}


def _parse_result(result: Any, use_timestamps: bool) -> tuple[str, List[Dict[str, Any]]]:
    """Parse NeMo transcription result."""
    text = ""
    segments: List[Dict[str, Any]] = []

    if not result:
        return text, segments

    # Extract text
    if isinstance(result, list) and result:
        first = result[0]
        if hasattr(first, "text"):
            text = str(first.text or "")
        elif isinstance(first, str):
            text = first
        else:
            text = str(first)

        # Extract segments/timestamps if available
        if use_timestamps and hasattr(first, "segments"):
            segments = _extract_segments(first.segments)
        elif use_timestamps and hasattr(first, "words"):
            segments = _extract_word_segments(first.words)
    elif isinstance(result, str):
        text = result

    return text.strip(), segments


def _extract_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    """Extract segments with timestamps."""
    segments: List[Dict[str, Any]] = []

    if not raw_segments:
        return segments

    for seg in raw_segments:
        start = _get_attr(seg, "start_time", "start", "start_offset")
        end = _get_attr(seg, "end_time", "end", "end_offset")
        text = _get_attr(seg, "text", "content", default="")

        if start is not None and end is not None:
            segments.append({
                "start": float(start),
                "end": float(end),
                "text": str(text).strip(),
            })

    return segments


def _extract_word_segments(raw_words: Any) -> List[Dict[str, Any]]:
    """Extract word-level timestamps."""
    segments: List[Dict[str, Any]] = []

    if not raw_words:
        return segments

    for word in raw_words:
        start = _get_attr(word, "start_time", "start_offset", "start")
        end = _get_attr(word, "end_time", "end_offset", "end")
        text = _get_attr(word, "text", "word", default="")

        if start is not None and end is not None:
            segments.append({
                "start": float(start),
                "end": float(end),
                "text": str(text).strip(),
            })

    return segments


def _get_attr(obj: Any, *keys: str, default: Any = None) -> Any:
    """Try multiple attribute/key names, return first found or default."""
    # Try dict access
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
    # Try object attributes
    else:
        for key in keys:
            if hasattr(obj, key):
                return getattr(obj, key)

    return default
