"""Model loading and inference helpers for NVIDIA Canary ASR."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo.collections.asr.models import ASRModel

_LOGGER = logging.getLogger(__name__)


class CanaryEngine:
    """Lazily loads the Canary model and performs transcription."""

    _instance: Optional["CanaryEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Optional[ASRModel] = None
        self._model_lock = threading.Lock()
        self._model_ready = threading.Event()

    @classmethod
    def instance(cls) -> "CanaryEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def is_ready(self) -> bool:
        return self._model_ready.is_set()

    def _load_model(self) -> ASRModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    _LOGGER.info("Loading NVIDIA Canary model 'nvidia/canary-1b-v2' on CPU…")
                    model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
                    model.to("cpu")
                    model.eval()
                    self._model = model
                    self._model_ready.set()
        return self._model

    def get_model(self) -> ASRModel:
        return self._load_model()

    def transcribe(
        self,
        audio_path: Path,
        source_lang: str,
        target_lang: str,
        timestamps: bool,
        beam_size: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run transcription and return text plus optional segments."""

        model = self._load_model()

        kwargs: Dict[str, Any] = {}
        if source_lang:
            kwargs["source_lang"] = source_lang
        if target_lang:
            kwargs["target_lang"] = target_lang
        if beam_size is not None:
            kwargs["beam_size"] = beam_size
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        use_hypotheses = timestamps
        if timestamps:
            kwargs["return_hypotheses"] = True

        result: Any = None
        attempt_kwargs = dict(kwargs)

        while True:
            try:
                result = model.transcribe([str(audio_path)], **attempt_kwargs)
                break
            except TypeError as exc:
                if not attempt_kwargs:
                    raise
                key, _ = attempt_kwargs.popitem()
                _LOGGER.warning("Model.transcribe rejected argument '%s': %s", key, exc)
                if key == "return_hypotheses":
                    use_hypotheses = False
            except Exception:
                _LOGGER.exception("Canary transcription failed")
                raise

        text, segments = _parse_transcription_output(result, use_hypotheses)
        return {"text": text, "segments": segments}


def _parse_transcription_output(result: Any, use_hypotheses: bool) -> tuple[str, List[Dict[str, Any]]]:
    text: str = ""
    segments: List[Dict[str, Any]] = []

    if isinstance(result, list) and result:
        first = result[0]
        if hasattr(first, "text"):
            text = getattr(first, "text") or ""
        elif isinstance(first, str):
            text = first
        else:
            text = str(first)

        if use_hypotheses and hasattr(first, "segments"):
            raw_segments = getattr(first, "segments") or []
            segments = _convert_segments(raw_segments)
        elif use_hypotheses and hasattr(first, "words"):
            raw_segments = getattr(first, "words") or []
            segments = _convert_word_segments(raw_segments)
        elif isinstance(first, str):
            text = first
    elif isinstance(result, str):
        text = result

    text = text.strip()

    if not segments:
        segments = []

    return text, segments


def _convert_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for seg in raw_segments:
        start = _safe_get(seg, "start_time", "start", "start_offset")
        end = _safe_get(seg, "end_time", "end", "end_offset")
        content = _safe_get(seg, "text", "content", default="")
        if start is None or end is None:
            continue
        converted.append(
            {
                "start": float(start),
                "end": float(end),
                "text": str(content).strip(),
            }
        )
    return converted


def _convert_word_segments(raw_segments: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for word in raw_segments:
        start = _safe_get(word, "start_time", "start_offset", "start")
        end = _safe_get(word, "end_time", "end_offset", "end")
        token = _safe_get(word, "text", "word", default="")
        if start is None or end is None:
            continue
        converted.append(
            {
                "start": float(start),
                "end": float(end),
                "text": str(token).strip(),
            }
        )
    return converted


def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
    else:
        for key in keys:
            if hasattr(obj, key):
                return getattr(obj, key)
    return default


def load_engine() -> CanaryEngine:
    return CanaryEngine.instance()
