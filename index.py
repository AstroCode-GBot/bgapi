from __future__ import annotations

import io
import json
import logging
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
from fastapi import Body, FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("background-remover-api")
MAX_UPLOAD_BYTES: Final[int] = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))
ALLOWED_MIME_TYPES: Final[set[str]] = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_PIL_FORMATS: Final[set[str]] = {"JPEG", "PNG", "WEBP"}
HEX_COLOR_RE: Final[re.Pattern[str]] = re.compile(r"^#[0-9a-fA-F]{6}$")
MODEL_FILENAME: Final[str] = "model_fp16.onnx"
MODEL_URL: Final[str] = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model_fp16.onnx?download=true",
)
# Use MODEL_DIR when a persistent disk is available (for example on Render).
# On serverless platforms, /tmp is the writable cache directory.
DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/background-remover-models"))
MODEL_PATH: Final[Path] = DEFAULT_MODEL_DIR / MODEL_FILENAME
MODEL_SIZE_BYTES: Final[int] = 490 * 1024 * 1024
MODEL_INPUT_SIZE: Final[tuple[int, int]] = (1024, 1024)
_session: ort.InferenceSession | None = None
_model_lock = __import__("threading").Lock()

app = FastAPI(
    title="Background Remover API",
    version="2.0.0",
    description="BiRefNet FP16 background removal API with automatic model download and caching.",
)


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "error": message})


def read_upload(upload: UploadFile) -> bytes | None:
    data = bytearray()
    while True:
        chunk = upload.file.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            return None
    return bytes(data)


def validate_image(data: bytes, content_type: str | None, strict_mime: bool = True) -> bool:
    if strict_mime and content_type not in ALLOWED_MIME_TYPES:
        return False
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            return image.format in ALLOWED_PIL_FORMATS
    except (UnidentifiedImageError, OSError):
        return False


def parse_hex_color(value: str) -> tuple[int, int, int]:
    if not HEX_COLOR_RE.fullmatch(value):
        raise ValueError("background_color must be a six-digit hex color such as #ffffff")
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def apply_background(image: Image.Image, mode: str, color: str) -> bytes:
    if mode not in {"transparent", "white", "color"}:
        raise ValueError("background must be transparent, white, or color")
    rgba = image.convert("RGBA")
    if mode == "transparent":
        output_image = rgba
    else:
        rgb = (255, 255, 255) if mode == "white" else parse_hex_color(color)
        output_image = Image.alpha_composite(
            Image.new("RGBA", rgba.size, (*rgb, 255)), rgba
        ).convert("RGBA")
    output = io.BytesIO()
    output_image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def ensure_model() -> Path:
    """Download the BiRefNet FP16 ONNX model once if it is not cached."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 400 * 1024 * 1024:
        return MODEL_PATH

    temp_path = MODEL_PATH.with_suffix(".download")
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass

    LOGGER.info("Downloading BiRefNet FP16 model from Hugging Face...")
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "background-remover-api/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (64 * 1024 * 1024) < len(chunk):
                    LOGGER.info("BiRefNet download: %.0f%%", downloaded * 100 / total)
        if downloaded < 400 * 1024 * 1024:
            raise RuntimeError(f"Downloaded model is unexpectedly small: {downloaded} bytes")
        temp_path.replace(MODEL_PATH)
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise

    return MODEL_PATH


def get_session() -> ort.InferenceSession:
    global _session
    if _session is not None:
        return _session
    with _model_lock:
        if _session is not None:
            return _session
        model_path = ensure_model()
        LOGGER.info("Loading BiRefNet ONNX model: %s", model_path)
        _session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        LOGGER.info(
            "BiRefNet loaded. input=%s output=%s",
            [(x.name, x.shape, x.type) for x in _session.get_inputs()],
            [(x.name, x.shape, x.type) for x in _session.get_outputs()],
        )
    return _session


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def remove_background(image_bytes: bytes, mode: str, color: str) -> bytes:
    session = get_session()
    with Image.open(io.BytesIO(image_bytes)) as source:
        original = source.convert("RGB")
        original_size = original.size

        # BiRefNet official processor: resize to 1024x1024 and ImageNet normalization.
        resized = original.resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = (
            pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = np.transpose(pixels, (2, 0, 1))[None, ...]

        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        # The FP16 export can accept float16; use the graph's declared input type.
        if "float16" in input_meta.type.lower():
            tensor = tensor.astype(np.float16)
        else:
            tensor = tensor.astype(np.float32)

        output_meta = session.get_outputs()[0]
        prediction = session.run([output_meta.name], {input_name: tensor})[0]
        prediction = np.asarray(prediction)

        # Official BiRefNet ONNX usage applies sigmoid to output_image.
        if prediction.ndim == 4:
            prediction = prediction[0, 0]
        elif prediction.ndim == 3:
            prediction = prediction[0]
        prediction = _sigmoid(prediction)
        prediction = np.clip(prediction, 0.0, 1.0)

        mask = Image.fromarray((prediction * 255.0).astype(np.uint8), mode="L")
        mask = mask.resize(original_size, Image.Resampling.BILINEAR)

        cutout = original.convert("RGBA")
        cutout.putalpha(mask)
        return apply_background(cutout, mode, color)


def validate_request_options(background: str, background_color: str) -> JSONResponse | None:
    if background not in {"transparent", "white", "color"}:
        return error_response("Invalid background mode", 400)
    if background == "color" and not HEX_COLOR_RE.fullmatch(background_color):
        return error_response("Invalid background color", 400)
    return None


def zip_results(results: list[tuple[str, bytes]]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for filename, content in results:
            output.writestr(filename, content)
    return archive.getvalue()


def download_image_url(image_url: str) -> tuple[bytes, str | None]:
    if not image_url.startswith(("http://", "https://")):
        raise ValueError("image_url must start with http:// or https://")
    request = urllib.request.Request(image_url, headers={"User-Agent": "background-remover-api/1.3"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        data = response.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise OverflowError("Image file is too large")
    return data, content_type


def download_telegram_file(file_id: str, bot_token: str) -> tuple[bytes, str | None]:
    if not file_id or not bot_token:
        raise ValueError("file_id and bot_token are required")
    api_request = urllib.request.Request(
        "https://api.telegram.org/bot" + bot_token + "/getFile?file_id=" + file_id,
        headers={"User-Agent": "background-remover-api/1.3"},
    )
    with urllib.request.urlopen(api_request, timeout=30) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    if not metadata.get("ok") or not metadata.get("result", {}).get("file_path"):
        raise ValueError("Telegram getFile failed")
    file_path = str(metadata["result"]["file_path"])
    return download_image_url("https://api.telegram.org/file/bot" + bot_token + "/" + file_path)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "success": True,
        "name": "Background Remover API",
        "version": "2.0.0",
        "status": "online",
        "background_modes": ["transparent", "white", "color"],
        "requires_api_key": False,
        "model": "BiRefNet FP16 1024",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {"success": True, "status": "healthy", "requires_api_key": False}


@app.post("/remove-bg", response_class=Response)
def remove_bg(
    image: UploadFile | None = File(default=None),
    background: str = Form(default="transparent"),
    background_color: str = Form(default="#ffffff"),
) -> Response:
    if image is None:
        return error_response("No image uploaded", 400)
    if image.content_type not in ALLOWED_MIME_TYPES:
        return error_response("Unsupported image format", 415)
    options_error = validate_request_options(background, background_color)
    if options_error:
        return options_error
    try:
        image_bytes = read_upload(image)
    finally:
        image.file.close()
    if image_bytes is None:
        return error_response("Image file is too large", 413)
    if not image_bytes or not validate_image(image_bytes, image.content_type):
        return error_response("Invalid image file", 400)
    try:
        result = remove_background(image_bytes, background, background_color)
    except Exception:
        LOGGER.exception("Background removal failed")
        return error_response("Background removal failed", 502)
    return Response(
        content=result,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="no-background-{background}.png"'},
    )


@app.get("/remove-bg", response_class=Response)
def remove_bg_simple(
    image_url: str = Query(..., description="Direct downloadable image URL"),
    background: str = Query(default="transparent"),
    background_color: str = Query(default="#ffffff"),
) -> Response:
    """Simplest integration: GET an image URL and receive a processed PNG."""
    return remove_bg_url(
        {"image_url": image_url, "background": background, "background_color": background_color}
    )


@app.post("/remove-bg-batch", response_class=Response)
def remove_bg_batch(
    images: list[UploadFile] = File(default=[]),
    background: str = Form(default="transparent"),
    background_color: str = Form(default="#ffffff"),
) -> Response:
    """Remove backgrounds from 1–10 images and return the PNG results as a ZIP archive."""
    if not images:
        return error_response("No images uploaded", 400)
    if len(images) > 10:
        return error_response("Too many images; maximum is 10", 400)
    options_error = validate_request_options(background, background_color)
    if options_error:
        return options_error

    results: list[tuple[str, bytes]] = []
    try:
        for index, image in enumerate(images, start=1):
            if image.content_type not in ALLOWED_MIME_TYPES:
                return error_response(f"Unsupported image format at index {index}", 415)
            image_bytes = read_upload(image)
            if image_bytes is None:
                return error_response(f"Image file is too large at index {index}", 413)
            if not image_bytes or not validate_image(image_bytes, image.content_type):
                return error_response(f"Invalid image file at index {index}", 400)
            result = remove_background(image_bytes, background, background_color)
            stem = Path(image.filename or f"image-{index}").stem
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip(".-") or f"image-{index}"
            results.append((f"{index:02d}-{safe_stem}-no-background.png", result))
    except Exception:
        LOGGER.exception("Batch background removal failed")
        return error_response("Background removal failed", 502)
    finally:
        for image in images:
            image.file.close()

    archive = zip_results(results)
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="background-removed-batch.zip"'},
    )


@app.post("/remove-bg-url", response_class=Response)
def remove_bg_url(payload: dict = Body(...)) -> Response:
    """Accept a downloadable image URL, useful for bot platforms that cannot upload multipart files."""
    image_url = str(payload.get("image_url", "")).strip()
    background = str(payload.get("background", "transparent"))
    background_color = str(payload.get("background_color", "#ffffff"))
    if not image_url:
        return error_response("image_url is required", 400)
    options_error = validate_request_options(background, background_color)
    if options_error:
        return options_error
    try:
        image_bytes, content_type = download_image_url(image_url)
        if not validate_image(image_bytes, content_type, strict_mime=False):
            return error_response("Invalid or unsupported image URL", 400)
        result = remove_background(image_bytes, background, background_color)
    except OverflowError:
        return error_response("Image file is too large", 413)
    except Exception:
        LOGGER.exception("URL background removal failed")
        return error_response("Could not download or process image URL", 502)
    return Response(
        content=result,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="no-background-{background}.png"'},
    )


@app.get("/remove-bg-url", response_class=Response)
def remove_bg_url_get(
    image_url: str = Query(...),
    background: str = Query(default="transparent"),
    background_color: str = Query(default="#ffffff"),
) -> Response:
    """GET form for Telegram bots that need a downloadable result URL."""
    return remove_bg_url(
        {"image_url": image_url, "background": background, "background_color": background_color}
    )


@app.get("/remove-bg-telegram", response_class=Response)
def remove_bg_telegram(
    file_id: str = Query(...),
    bot_token: str = Query(...),
    background: str = Query(default="transparent"),
    background_color: str = Query(default="#ffffff"),
) -> Response:
    """Resolve a Telegram file_id through getFile and return the processed PNG."""
    options_error = validate_request_options(background, background_color)
    if options_error:
        return options_error
    try:
        image_bytes, content_type = download_telegram_file(file_id, bot_token)
        if not validate_image(image_bytes, content_type, strict_mime=False):
            return error_response("Invalid or unsupported Telegram image", 400)
        result = remove_background(image_bytes, background, background_color)
    except OverflowError:
        return error_response("Image file is too large", 413)
    except Exception:
        LOGGER.exception("Telegram background removal failed")
        return error_response("Could not download or process Telegram image", 502)
    return Response(
        content=result,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="no-background-{background}.png"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
