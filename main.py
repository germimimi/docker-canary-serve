"""FastAPI service for NVIDIA Canary ASR - CPU inference."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from canary_api.engine import CanaryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("canary.api")

# Configuration
DEFAULT_SOURCE_LANG = os.getenv("SOURCE_LANG", "ru")
DEFAULT_TARGET_LANG = os.getenv("TARGET_LANG", "ru")
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE", str(200 * 1024 * 1024)))  # 200MB

# Supported languages (25 European languages from NVIDIA Canary 1B V2)
SUPPORTED_LANGUAGES = {
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt",
    "ro", "ru", "sk", "sl", "sv"
}

# Initialize engine
engine = CanaryEngine.instance()

# Create FastAPI app
app = FastAPI(
    title="NVIDIA Canary ASR API",
    version="2.0.0",
    description="CPU-based ASR service for m4a audio files using NVIDIA Canary 1B V2",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load model on startup."""
    LOGGER.info("Starting up: loading NVIDIA Canary model...")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, engine.get_model)
        LOGGER.info("Model loaded successfully and ready for inference.")
    except Exception as exc:
        LOGGER.exception("Failed to load model during startup")
        raise RuntimeError(f"Model initialization failed: {exc}") from exc


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="Model is still loading")
    return {"status": "ok"}


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    source_lang: str = Form(DEFAULT_SOURCE_LANG),
    target_lang: str = Form(DEFAULT_TARGET_LANG),
    timestamps: bool = Form(False),
) -> JSONResponse:
    """
    Transcribe m4a audio file.

    Args:
        file: Audio file in m4a format
        source_lang: Source language (ISO 639-1 code)
        target_lang: Target language (ISO 639-1 code)
        timestamps: Whether to include timestamps in response

    Returns:
        JSON with transcription text and optional segments
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    extension = Path(file.filename).suffix.lower()
    if extension != ".m4a":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{extension}'. Only .m4a files are supported.",
        )

    # Validate languages
    if source_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source language '{source_lang}'. Supported: {sorted(SUPPORTED_LANGUAGES)}",
        )

    if target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language '{target_lang}'. Supported: {sorted(SUPPORTED_LANGUAGES)}",
        )

    # Process file
    with tempfile.TemporaryDirectory(prefix="canary_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        input_path = tmp_dir_path / f"input{extension}"

        # Save uploaded file
        file_size = await _save_upload_to_disk(file, input_path)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
            )

        # Convert m4a to WAV (16kHz mono)
        wav_path = await _convert_to_wav(input_path, tmp_dir_path)

        # Run transcription (non-blocking)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                engine.transcribe,
                wav_path,
                source_lang,
                target_lang,
                timestamps,
            )
        except Exception as exc:
            LOGGER.exception("Transcription failed")
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please try again.",
            ) from exc

    # Build response
    text = result.get("text", "").strip()
    response: Dict[str, Any] = {"text": text}

    if timestamps:
        segments = result.get("segments", [])
        response["segments"] = segments

    return JSONResponse(content=response)


async def _save_upload_to_disk(upload: UploadFile, destination: Path) -> int:
    """Save uploaded file to disk and return size in bytes."""
    size = 0
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            size += len(chunk)
            buffer.write(chunk)
    await upload.close()
    return size


async def _convert_to_wav(input_path: Path, tmp_dir_path: Path) -> Path:
    """Convert audio to WAV format (16kHz mono) using ffmpeg."""
    output_path = tmp_dir_path / "converted.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16kHz
        str(output_path),
    ]

    LOGGER.info("Converting audio: %s", " ".join(command))

    loop = asyncio.get_running_loop()
    try:
        completed = await loop.run_in_executor(
            None,
            lambda: subprocess.run(command, capture_output=True, check=True)
        )
        if completed.stderr:
            LOGGER.debug("ffmpeg stderr: %s", completed.stderr.decode(errors="ignore"))
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Audio conversion failed: %s", exc.stderr.decode(errors="ignore"))
        raise HTTPException(
            status_code=500,
            detail="Audio conversion failed. Please ensure the file is a valid m4a audio file.",
        ) from exc

    return output_path


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "9000")),
    )
