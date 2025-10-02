from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from canary_api.engine import load_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("canary.api")

DEFAULT_SOURCE_LANG = os.getenv("SOURCE_LANG", "ru")
DEFAULT_TARGET_LANG = os.getenv("TARGET_LANG", "ru")
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE", str(200 * 1024 * 1024)))
SUPPORTED_EXTENSIONS = {".wav", ".m4a", ".flac"}
MODEL_NAME = "nvidia/canary-1b-v2"

engine = load_engine()

app = FastAPI(
    title="NVIDIA Canary 1B V2 ASR API",
    version="2.0.0",
    description="FastAPI server for CPU inference with NVIDIA Canary 1B V2",
)


@app.on_event("startup")
async def startup_event() -> None:
    LOGGER.info("Application startup: ensuring model is loaded.")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, engine.get_model)
    LOGGER.info("Model loaded and ready for inference.")


@app.get("/health")
async def health() -> Dict[str, str]:
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="Model is still loading")
    return {"status": "ok"}


@app.post("/inference")
async def run_inference(
    file: UploadFile = File(...),
    source_lang: str = Form(DEFAULT_SOURCE_LANG),
    target_lang: str = Form(DEFAULT_TARGET_LANG),
    timestamps: str = Form("no"),
    response_format: str = Form("json"),
    beam_size: int = Form(1),
    batch_size: int = Form(1),
) -> Response:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    timestamps_requested = _parse_bool(timestamps)
    response_format_normalized = response_format.strip().lower()
    if response_format_normalized not in {"json", "verbose_json", "text", "srt", "vtt"}:
        raise HTTPException(status_code=400, detail="Unsupported response_format value")

    need_segments = timestamps_requested or response_format_normalized in {"srt", "vtt"}

    with tempfile.TemporaryDirectory(prefix="canary_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / f"input{extension}"
        file_size = await _save_upload_to_disk(file, input_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")

        processed_path = _prepare_audio(input_path, extension, tmp_dir_path)

        try:
            result = engine.transcribe(
                processed_path,
                source_lang=source_lang,
                target_lang=target_lang,
                timestamps=need_segments,
                beam_size=beam_size,
                batch_size=batch_size,
            )
        except Exception as exc:  # pragma: no cover - propagated to client
            LOGGER.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    text = result.get("text", "").strip()
    segments = result.get("segments", []) if need_segments else []

    if response_format_normalized == "json":
        payload: Dict[str, object] = {"text": text}
        if timestamps_requested:
            payload["segments"] = segments
        return JSONResponse(content=payload)

    if response_format_normalized == "verbose_json":
        verbose_payload = {
            "text": text,
            "model": MODEL_NAME,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "segments": segments,
        }
        return JSONResponse(content=verbose_payload)

    if response_format_normalized == "text":
        return PlainTextResponse(content=text)

    formatted_segments = segments if segments else [{"start": 0.0, "end": 0.0, "text": text}]

    if response_format_normalized == "srt":
        return PlainTextResponse(content=_segments_to_srt(formatted_segments), media_type="application/x-subrip")

    if response_format_normalized == "vtt":
        return PlainTextResponse(content=_segments_to_vtt(formatted_segments), media_type="text/vtt")

    raise HTTPException(status_code=500, detail="Unhandled response format")


async def _save_upload_to_disk(upload: UploadFile, destination: Path) -> int:
    size = 0
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            buffer.write(chunk)
    await upload.close()
    return size


def _prepare_audio(input_path: Path, extension: str, tmp_dir_path: Path) -> Path:
    if extension != ".wav":
        return _convert_to_wav(input_path, tmp_dir_path)

    try:
        info = sf.info(str(input_path))
        if info.samplerate == 16000 and info.channels == 1:
            return input_path
        LOGGER.info("Re-sampling WAV file from %s Hz / %s channels", info.samplerate, info.channels)
    except RuntimeError:
        LOGGER.info("soundfile could not read WAV metadata, forcing conversion")

    return _convert_to_wav(input_path, tmp_dir_path)


def _convert_to_wav(input_path: Path, tmp_dir_path: Path) -> Path:
    output_path = tmp_dir_path / "converted.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    LOGGER.info("Running ffmpeg conversion: %s", " ".join(command))
    try:
        completed = subprocess.run(command, capture_output=True, check=True)
        if completed.stderr:
            LOGGER.debug("ffmpeg stderr: %s", completed.stderr.decode(errors="ignore"))
    except subprocess.CalledProcessError as exc:
        LOGGER.error("ffmpeg conversion failed: %s", exc.stderr.decode(errors="ignore"))
        raise HTTPException(status_code=500, detail="Failed to convert audio to WAV") from exc
    return output_path


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _format_timestamp(seconds: float, separator: str) -> str:
    total_ms = max(seconds, 0.0) * 1000.0
    hours = int(total_ms // 3_600_000)
    minutes = int((total_ms % 3_600_000) // 60_000)
    secs = (total_ms % 60_000) / 1000.0
    if separator == ",":
        return f"{hours:02}:{minutes:02}:{secs:06.3f}".replace(".", ",")
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"


def _segments_to_srt(segments: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        lines.append(str(idx))
        lines.append(f"{_format_timestamp(start, ',')} --> {_format_timestamp(end, ',')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _segments_to_vtt(segments: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        lines.append(f"{_format_timestamp(start, '.')} --> {_format_timestamp(end, '.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("APP_HOST", "0.0.0.0"), port=int(os.getenv("APP_PORT", "9000")))
