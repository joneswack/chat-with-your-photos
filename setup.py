import base64
import csv
import io
import re
import sys
from pathlib import Path

import pycountry
import reverse_geocoder as rg
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel
from tqdm import tqdm

from utils import (
    anthropic_models,
    ollama_models,
    openai_models,
    pick,
    register_folder,
)

load_dotenv()
register_heif_opener()

# EXIF tag IDs
_EXIF_DATE_TAKEN = 36867  # DateTimeOriginal — when the shutter fired
_EXIF_DATE_MODIFIED = 306  # DateTime — file modification time (fallback)
_EXIF_GPS_IFD = 34853

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

FIELDNAMES = [
    "relative_file_path",
    "label",
    "description",
    "timestamp",
    "lat",
    "lon",
    "country",
    "region",
    "city",
]


class ImageDescription(BaseModel):
    label: str
    description: str


def get_timestamp(exif) -> str:
    """Return the capture timestamp string from EXIF data, or empty string if absent."""
    for tag in (_EXIF_DATE_TAKEN, _EXIF_DATE_MODIFIED):
        val = exif.get(tag)
        if val:
            return val
    return ""


def get_gps_coords(exif) -> tuple[float, float] | None:
    """Return (latitude, longitude) in decimal degrees from EXIF GPS data, or None."""
    gps_ifd = exif.get_ifd(_EXIF_GPS_IFD)
    if not gps_ifd:
        return None
    lat_ref = gps_ifd.get(1)
    lat = gps_ifd.get(2)
    lon_ref = gps_ifd.get(3)
    lon = gps_ifd.get(4)
    if not all([lat_ref, lat, lon_ref, lon]):
        return None

    def dms_to_dd(dms):
        d, m, s = dms
        return float(d) + float(m) / 60 + float(s) / 3600

    return (
        dms_to_dd(lat) * (-1 if lat_ref == "S" else 1),
        dms_to_dd(lon) * (-1 if lon_ref == "W" else 1),
    )


def encode_image(img: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_llm_json(text: str) -> ImageDescription:
    """Extract JSON from LLM response, tolerating markdown code fences."""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    return ImageDescription.model_validate_json(raw)


def label_with_llm(
    img: Image.Image, location: str, provider: str, model_name: str
) -> tuple[str, str]:
    """Ask an LLM to produce a (label, description) pair for the given image."""
    location_hint = (
        f" The photo was taken in {location}, which can help you label the image more precisely."
        if location
        else ""
    )
    prompt = (
        f"Extract a label (max. 3 words) and a short one-sentence description from the photo."
        f"{location_hint} Respond with JSON only: "
        '{"label": "...", "description": "..."}'
    )

    if provider == "ollama":
        from ollama import chat

        response = chat(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt, "images": [encode_image(img)]}
            ],
            think=False,
            format=ImageDescription.model_json_schema(),
            options={"temperature": 0},
        )
        desc = ImageDescription.model_validate_json(
            response.message.content or '{"label": "", "description": ""}'
        )
        return desc.label, desc.description

    b64 = encode_image(img)

    if provider == "anthropic":
        import anthropic
        from anthropic.types import TextBlock

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text_block = next(b for b in msg.content if isinstance(b, TextBlock))
        desc = _parse_llm_json(text_block.text)
        return desc.label, desc.description

    if provider == "openai":
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        desc = _parse_llm_json(response.choices[0].message.content or "")
        return desc.label, desc.description

    return "", ""


def _ask_folder() -> Path:
    """Prompt until the user enters a valid directory path."""
    while True:
        raw = input("Full path to photos folder: ").strip()
        if not raw:
            continue
        folder = Path(raw).expanduser().resolve()
        if folder.is_dir():
            return folder
        print(f"  Not a directory: {folder}")


def _pick_model() -> tuple[str, str, str]:
    """Present the model picker and return (provider, model_name, display_label).

    timm is pure image classification (no token generation), so it's orders of
    magnitude faster than LLM captioning — the recommended default for large collections.
    """
    models: dict[str, tuple[str, str]] = {
        "ViT Image Classifier  (Local · label classification · very fast and recommended for 100+ photos)": (
            "timm",
            "timm",
        ),
    }
    for name in ollama_models():
        models[f"{name}  (ollama · LLM captioning)"] = ("ollama", name)
    for mid in anthropic_models():
        models[f"{mid}  (anthropic · LLM captioning)"] = ("anthropic", mid)
    for mid in openai_models():
        models[f"{mid}  (openai · LLM captioning)"] = ("openai", mid)

    model_key = pick("Select vision model for labeling:", list(models))
    provider, model_name = models[model_key]
    return provider, model_name, model_key.split("(")[0].strip()


def _collect_image_paths(images_dir: Path) -> list[Path]:
    """Return all supported images under images_dir, excluding the .cwyp index folder."""
    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.suffix.lower() in _IMAGE_EXTENSIONS and ".cwyp" not in p.parts
    )


def _extract_metadata(
    image_paths: list[Path],
) -> tuple[list[str], list[tuple[float, float] | None]]:
    """Fast EXIF pass — return parallel lists of timestamps and GPS coords."""
    print("Reading EXIF metadata...")
    timestamps: list[str] = []
    coords_list: list[tuple[float, float] | None] = []
    for path in tqdm(image_paths, unit="img"):
        exif = Image.open(path).getexif()
        timestamps.append(get_timestamp(exif))
        coords_list.append(get_gps_coords(exif))
    return timestamps, coords_list


def _reverse_geocode(
    coords_list: list[tuple[float, float] | None],
) -> list[dict | None]:
    """Batch reverse-geocode GPS coordinates; returns a parallel list of geo dicts."""
    geo_data: list[dict | None] = [None] * len(coords_list)
    gps_indices = [i for i, c in enumerate(coords_list) if c is not None]
    gps_coords = [c for c in coords_list if c is not None]
    if gps_indices:
        results = rg.search(gps_coords, verbose=False)
        for i, result in zip(gps_indices, results):
            country_obj = pycountry.countries.get(alpha_2=result["cc"])
            geo_data[i] = {
                "country": country_obj.name if country_obj else result["cc"],
                "region": result["admin1"],
                "city": result["name"],
            }
    return geo_data


def _label_images(
    image_paths: list[Path],
    images_dir: Path,
    timestamps: list[str],
    coords_list: list[tuple[float, float] | None],
    geo_data: list[dict | None],
    provider: str,
    model_name: str,
    model_label: str,
) -> list[dict[str, str]]:
    """Label every image and return the list of CSV rows."""
    if provider == "timm":
        import timm
        import timm.data
        import torch
        from timm.data.imagenet_info import ImageNetInfo

        timm_model = timm.create_model(
            "timm/vit_base_patch16_224.augreg_in21k", pretrained=True
        ).eval()
        config = timm.data.resolve_model_data_config(timm_model)
        transforms = timm.data.create_transform(**config, is_training=False)
        info = ImageNetInfo("imagenet21k")

    rows: list[dict[str, str]] = []
    print(f"Labeling with {model_label}...")
    for path, timestamp, coords, geo in tqdm(
        zip(image_paths, timestamps, coords_list, geo_data),
        total=len(image_paths),
        unit="img",
    ):
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

        if provider == "timm":
            with torch.no_grad():
                output = timm_model(transforms(img).unsqueeze(0))
            label = info.index_to_description(int(output.argmax(dim=1).item()))
            description = ""
        else:
            label, description = label_with_llm(
                img, geo["country"] if geo else "", provider, model_name
            )

        rows.append(
            {
                "relative_file_path": str(path.relative_to(images_dir)),
                "label": label,
                "description": description,
                "timestamp": timestamp,
                "lat": coords[0] if coords else "",
                "lon": coords[1] if coords else "",
                "country": geo["country"] if geo else "",
                "region": geo["region"] if geo else "",
                "city": geo["city"] if geo else "",
            }
        )
    return rows


def main() -> None:
    """Entry point when called from the top-level menu; wraps _run() with interrupt handling."""
    print("Press Ctrl+C or Ctrl+D to return to the menu.\n")
    try:
        _run()
    except (KeyboardInterrupt, EOFError):
        print("\nReturning to the menu.")


def _run() -> None:
    """Prompt the user for a folder and model, index all photos, and write the CSV."""
    images_dir = _ask_folder()
    output_csv = images_dir / ".cwyp" / "index.csv"
    provider, model_name, model_label = _pick_model()
    output_csv.parent.mkdir(exist_ok=True)

    image_paths = _collect_image_paths(images_dir)
    timestamps, coords_list = _extract_metadata(image_paths)
    geo_data = _reverse_geocode(coords_list)
    rows = _label_images(
        image_paths, images_dir, timestamps, coords_list, geo_data,
        provider, model_name, model_label,
    )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")
    register_folder(images_dir)


if __name__ == "__main__":
    try:
        _run()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        sys.exit(0)
