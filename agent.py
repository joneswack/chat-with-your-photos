import base64
import io
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pillow_heif import register_heif_opener

from utils import (
    anthropic_models,
    ollama_models,
    openai_models,
    pick,
)

load_dotenv()
register_heif_opener()

_llm: Any = None   # set in _run() once the user picks a model
CSV_PATH: Path     # set in _run() after the user picks a folder

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _images_dir() -> Path:
    """Return the root photo directory derived from the active CSV path."""
    return CSV_PATH.parent.parent.resolve()


def _validated_photo_path(relative_file_path: str) -> Path:
    """Resolve and validate a relative photo path; raise ValueError with a user-facing message on failure."""
    images_dir = _images_dir()
    full = (images_dir / relative_file_path).resolve()
    if not full.is_relative_to(images_dir):
        raise ValueError("Access denied: path is outside the photo collection folder.")
    if full.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{full.suffix}'.")
    if not full.exists():
        raise ValueError(f"File not found: {full}")
    return full


def _load() -> pd.DataFrame:
    """Load the index CSV and parse the timestamp column."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Run setup.py first.")
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%Y:%m:%d %H:%M:%S", errors="coerce"
    )
    return df


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

@tool
def get_collection_overview() -> str:
    """Return high-level stats: total photos, date range, countries covered."""
    df = _load()
    total = len(df)
    dated = df["timestamp"].dropna()
    date_range = (
        f"{dated.min().date()} → {dated.max().date()}" if not dated.empty else "unknown"
    )
    countries = df["country"].replace("", pd.NA).dropna().unique().tolist()
    return (
        f"Total photos: {total}\n"
        f"Date range: {date_range}\n"
        f"Countries: {', '.join(sorted(countries)) or 'none with GPS'}"
    )


@tool
def get_locations(group_by: str = "city") -> str:
    """
    Return photo counts grouped by location.

    group_by: 'country', 'region', or 'city' (default 'city')
    """
    if group_by not in ("country", "region", "city"):
        return "group_by must be 'country', 'region', or 'city'."
    df = _load()
    counts = (
        df[df[group_by].str.strip().astype(bool)]
        .groupby(group_by)
        .size()
        .sort_values(ascending=False)
    )
    if counts.empty:
        return f"No photos have {group_by} data."
    lines = [f"  {name}: {n} photos" for name, n in counts.items()]
    return f"Photos by {group_by}:\n" + "\n".join(lines)


@tool
def filter_photos(
    location: str = "",
    label_keyword: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 20,
    offset: int = 0,
) -> str:
    """
    Filter photos. All parameters are optional and combinable.

    location: substring to match against country, region, or city (case-insensitive)
    label_keyword: substring to match against the photo label (case-insensitive)
    date_from: start date as YYYY-MM-DD (inclusive)
    date_to:   end date as YYYY-MM-DD (inclusive)
    limit: max results to return (default 20)
    offset: number of results to skip, for pagination (default 0)

    Results are sorted by timestamp (oldest first). Photos without a timestamp appear last.
    Each result row: relative_file_path | YYYY-MM-DD HH:MM | city, region, country | label | description]
    """
    df = _load()
    mask = pd.Series(True, index=df.index)

    if location:
        loc = location.lower()
        mask &= (
            df["country"].str.lower().str.contains(loc, na=False)
            | df["region"].str.lower().str.contains(loc, na=False)
            | df["city"].str.lower().str.contains(loc, na=False)
        )
    if label_keyword:
        mask &= df["label"].str.lower().str.contains(label_keyword.lower(), na=False)
    if date_from:
        mask &= df["timestamp"] >= pd.Timestamp(date_from)
    if date_to:
        mask &= df["timestamp"] <= pd.Timestamp(date_to) + pd.Timedelta(days=1)

    filtered = df[mask].sort_values("timestamp", na_position="last")
    result = filtered.iloc[offset : offset + limit]
    if result.empty:
        return "No photos match the given filters."

    rows = []
    for _, row in result.iterrows():
        timestamp_str = (
            row["timestamp"].strftime("%Y-%m-%d %H:%M")
            if pd.notna(row["timestamp"])
            else "?"
        )
        place = ", ".join(
            v for v in [row["city"], row["region"], row["country"]]
            if isinstance(v, str) and v
        )
        desc = row["description"]
        desc_str = f" | {desc}" if isinstance(desc, str) and desc.strip() else ""
        rows.append(
            f"  {row['relative_file_path']} | {timestamp_str} | {place or 'no GPS'} | {row['label']}{desc_str}"
        )
    header = f"Found {len(filtered)} photos (showing {offset + 1}–{offset + len(result)}):"
    return header + "\n" + "\n".join(rows)


@tool
def get_label_distribution(top_n: int = 15, offset: int = 0) -> str:
    """
    Return the most common scene/object labels across the collection.

    top_n: number of labels to return (default 15)
    offset: number of labels to skip, for pagination (default 0)
    """
    df = _load()
    counts = df["label"].value_counts()
    total = len(counts)
    page = counts.iloc[offset : offset + top_n]
    if page.empty:
        return "No labels found."
    lines = [f"  {label}: {n}" for label, n in page.items()]
    header = f"Labels {offset + 1}–{offset + len(page)} of {total}:"
    return header + "\n" + "\n".join(lines)


@tool
def display_photo(relative_file_path: str) -> str:
    """
    Open a photo in the system's default image viewer.

    relative_file_path: the relative_file_path value from the index (e.g. '2024/IMG_1234.HEIC')
    """
    try:
        full_path = _validated_photo_path(relative_file_path)
    except ValueError as e:
        return str(e)
    subprocess.Popen(["open", str(full_path)])
    return f"Opened {relative_file_path}"


@tool
def describe_photo(relative_file_path: str) -> str:
    """
    Return metadata for a single photo. If no description is stored, a vision
    LLM analyses the image and writes the result back to the index.

    relative_file_path: the relative_file_path value from the index

    Output fields:
      File        — relative_file_path
      Label       — short scene/object label
      Description — one-sentence description (generated on first call if missing)
      Timestamp   — YYYY-MM-DD HH:MM, or 'unknown'
      GPS         — decimal lat, lon, or 'no GPS data'
      Location    — city, region, country (reverse-geocoded), or 'no GPS data'
    """
    from PIL import Image

    df = _load()
    mask = df["relative_file_path"] == relative_file_path
    if not mask.any():
        return f"Photo not found in index: {relative_file_path}"

    row = df[mask].iloc[0]
    description = row["description"]

    if not isinstance(description, str) or not description.strip():
        try:
            full = _validated_photo_path(relative_file_path)
        except ValueError as e:
            return str(e)

        buf = io.BytesIO()
        Image.open(full).convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        location = ", ".join(
            v for v in [row["city"], row["country"]] if isinstance(v, str) and v.strip()
        )
        location_hint = (
            f" The photo was taken in {location}, which can help you describe it more precisely."
            if location
            else ""
        )
        reply = _llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": f"Describe this photo in one concise sentence.{location_hint}",
                        },
                    ]
                )
            ]
        )
        description = reply.content
        if isinstance(description, list):
            description = " ".join(
                b["text"]
                for b in description
                if isinstance(b, dict) and b.get("type") == "text"
            )

        raw = pd.read_csv(CSV_PATH)
        raw["description"] = raw["description"].astype(object)
        raw.loc[raw["relative_file_path"] == relative_file_path, "description"] = description
        raw.to_csv(CSV_PATH, index=False)

    ts = row["timestamp"]
    timestamp_str = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "unknown"

    lat = pd.to_numeric(row.get("lat", ""), errors="coerce")
    lon = pd.to_numeric(row.get("lon", ""), errors="coerce")
    gps_str = (
        f"{lat:.6f}, {lon:.6f}" if pd.notna(lat) and pd.notna(lon) else "no GPS data"
    )
    location = (
        ", ".join(
            v for v in [row["city"], row["region"], row["country"]]
            if isinstance(v, str) and v.strip()
        )
        or "no GPS data"
    )

    return (
        f"File:        {relative_file_path}\n"
        f"Label:       {row['label']}\n"
        f"Description: {description or '(none)'}\n"
        f"Timestamp:   {timestamp_str}\n"
        f"GPS:         {gps_str}\n"
        f"Location:    {location}"
    )


@tool
def generate_image(
    prompt: str,
    input_image_paths: list[str] | None = None,
) -> str:
    """
    Generate an image from a text description and open it for viewing.

    prompt: what to draw (e.g. 'a sunset over the Serengeti with elephants')
    input_image_paths: up to 3 relative_file_path values from the index to use as
                       visual reference (optional)
    """
    import tempfile
    import threading

    from langchain_openai import ChatOpenAI
    from PIL import Image

    if not isinstance(_llm, ChatOpenAI):
        return (
            "Image generation requires an OpenAI GPT model. "
            "Please restart and select a GPT model to use this feature."
        )

    def _load_as_base64_jpeg(rel_path: str) -> str:
        """Load a photo and return it as a base64 JPEG string."""
        full = _validated_photo_path(rel_path)
        buf = io.BytesIO()
        Image.open(full).convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    content: list = [{"type": "text", "text": prompt}]
    for rel in (input_image_paths or [])[:3]:
        try:
            b64 = _load_as_base64_jpeg(rel)
        except (ValueError, FileNotFoundError) as e:
            return str(e)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    llm = _llm.bind_tools([{"type": "image_generation", "quality": "high"}])
    ai_msg = llm.invoke([{"role": "user", "content": content}])
    image_block = next((b for b in ai_msg.content_blocks if b["type"] == "image"), None)
    if image_block is None:
        return "Image generation returned no image."

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="cwyp_gen_") as f:
        f.write(base64.b64decode(image_block["base64"]))
        out_path = f.name

    def _open_and_cleanup(path: str) -> None:
        subprocess.run(["open", "-W", path])
        Path(path).unlink(missing_ok=True)

    threading.Thread(target=_open_and_cleanup, args=(out_path,), daemon=True).start()
    return "Generated image opened (file will be removed when the viewer is closed)."


@tool
def get_photos_by_date(date: str) -> str:
    """
    Return all photos taken on a specific date.

    date: YYYY-MM-DD
    """
    df = _load()
    day = pd.Timestamp(date).date()
    result = df[df["timestamp"].dt.date == day]
    if result.empty:
        return f"No photos found for {date}."
    rows = []
    for _, row in result.iterrows():
        place = ", ".join(
            v for v in [row["city"], row["region"], row["country"]]
            if isinstance(v, str) and v
        )
        rows.append(
            f"  {row['relative_file_path']} | {place or 'no GPS'} | {row['label']}"
        )
    return f"{len(result)} photos on {date}:\n" + "\n".join(rows)


@tool
def create_trajectory_map(
    location: str = "",
    date_from: str = "",
    date_to: str = "",
) -> str:
    """
    Generate an HTML map showing the photo trajectory (path + markers) and open it in the browser.

    location: optional substring filter on country, region, or city
    date_from: optional start date as YYYY-MM-DD
    date_to:   optional end date as YYYY-MM-DD

    Only photos with GPS coordinates are included. Photos are connected in chronological order.
    """
    import tempfile

    import folium
    from PIL import Image

    def _thumbnail_b64(rel_path: str, width: int = 220) -> str | None:
        """Return a resized base64 JPEG thumbnail, or None if the image can't be loaded."""
        try:
            full = _validated_photo_path(rel_path)
            img = Image.open(full).convert("RGB")
            ratio = width / img.width
            img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    df = _load()
    df = df[df["lat"].notna() & df["lon"].notna() & (df["lat"] != "") & (df["lon"] != "")]
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    if location:
        loc = location.lower()
        df = df[
            df["country"].str.lower().str.contains(loc, na=False)
            | df["region"].str.lower().str.contains(loc, na=False)
            | df["city"].str.lower().str.contains(loc, na=False)
        ]
    if date_from:
        df = df[df["timestamp"] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df["timestamp"] <= pd.Timestamp(date_to) + pd.Timedelta(days=1)]

    df = df.sort_values("timestamp", na_position="last")
    if df.empty:
        return "No photos with GPS coordinates match the given filters."

    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=7)
    folium.PolyLine(
        list(zip(df["lat"], df["lon"])), color="#E74C3C", weight=2.5, opacity=0.8
    ).add_to(m)

    for _, row in df.iterrows():
        ts = (
            row["timestamp"].strftime("%Y-%m-%d %H:%M")
            if pd.notna(row["timestamp"])
            else "?"
        )
        place = ", ".join(
            v for v in [row["city"], row["country"]] if isinstance(v, str) and v
        )
        b64 = _thumbnail_b64(row["relative_file_path"])
        img_tag = (
            f'<img src="data:image/jpeg;base64,{b64}" style="width:220px;display:block;margin-bottom:4px;">'
            if b64
            else ""
        )
        popup_html = (
            f"{img_tag}"
            f"<b>{row['label']}</b><br>"
            f'<span style="color:#555;font-size:0.85em">{ts} &middot; {place}</span>'
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color="#2C3E50",
            fill=True,
            fill_color="#3498DB",
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=row["label"],
        ).add_to(m)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, prefix="cwyp_map_") as f:
        out_path = f.name
    m.save(out_path)
    subprocess.Popen(["open", out_path])
    return f"Trajectory map opened ({len(df)} photos with GPS)."


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _pick_model() -> str | None:
    """Build the model menu, print provider tips, and return the selected 'provider:model_id' string."""
    models: dict[str, str] = {}
    for model_id in anthropic_models():
        models[f"{model_id} (anthropic)"] = f"anthropic:{model_id}"
    if not any(v.startswith("anthropic:") for v in models.values()):
        print("Tip: set ANTHROPIC_API_KEY in .env to see available Claude models.\n")
    for model_id in openai_models():
        models[f"{model_id} (openai)"] = f"openai:{model_id}"
    if not any(v.startswith("openai:") for v in models.values()):
        print("Tip: set OPENAI_API_KEY in .env to see available OpenAI models.\n")
    for name in ollama_models():
        models[f"{name} (ollama)"] = f"ollama:{name}"
    if not any(v.startswith("ollama:") for v in models.values()):
        print("Tip: install Ollama (https://ollama.com) to use local models.\n")
    if not models:
        print("No models available. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or install Ollama.")
        return None
    return models[pick("Select model:", list(models))]


def _build_llm(model_str: str) -> Any:
    """Instantiate and return the LangChain chat model for the given 'provider:model_id' string."""
    if model_str.startswith("ollama:"):
        return ChatOllama(model=model_str.removeprefix("ollama:"))
    if model_str.startswith("anthropic:"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model_name=model_str.removeprefix("anthropic:"))
    if model_str.startswith("openai:"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model_name=model_str.removeprefix("openai:"))
    return model_str


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main(folder: Path) -> None:
    """Entry point when called from the top-level menu; wraps _run() with interrupt handling."""
    print("Press Ctrl+C or Ctrl+D at any time to return to the menu.\n")
    try:
        _run(folder)
    except (KeyboardInterrupt, EOFError):
        print("\nReturning to the menu.")


def _run(folder: Path) -> None:
    """Pick a model, create the agent, and run the chat REPL."""
    global CSV_PATH, _llm
    CSV_PATH = folder / ".cwyp" / "index.csv"

    model_str = _pick_model()
    if model_str is None:
        return
    _llm = _build_llm(model_str)
    debug = pick("Show tool calls?", ["off", "on"]) == "on"

    agent = create_agent(
        model=_llm,
        tools=[
            get_collection_overview,
            get_locations,
            filter_photos,
            get_label_distribution,
            get_photos_by_date,
            display_photo,
            describe_photo,
            generate_image,
            create_trajectory_map,
        ],
        system_prompt=(
            "Answer the user query using the available tools. "
            "When looking for photos with specific labels, first check the label distribution. "
            "If the precise label does not exist, check for similar alternatives."
        ),
    )

    photo_count = len(pd.read_csv(CSV_PATH, usecols=["relative_file_path"]))
    print(
        f"Your selected folder has {photo_count:,} photos indexed.\n"
        f"Ask me anything — e.g. 'What places did I visit?' or 'Show me photos of animals'.\n"
        f"Press Ctrl+C or Ctrl+D to return to the menu.\n"
    )

    history = []
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))

        new_messages: list = []
        printed_prefix = False
        seen_tcs: set[str] = set()
        for chunk in agent.stream(
            {"messages": history},
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, _ = chunk["data"]
                if (
                    isinstance(token, AIMessageChunk)
                    and token.text
                    and not token.tool_call_chunks
                ):
                    if not printed_prefix:
                        print("\nAgent: ", end="", flush=True)
                        printed_prefix = True
                    print(token.text, end="", flush=True)
            elif chunk["type"] == "updates":
                for source, update in chunk["data"].items():
                    if source in ("model", "tools"):
                        new_messages.extend(update.get("messages", []))
                    if debug and source == "model":
                        msg = update["messages"][-1]
                        for tc in getattr(msg, "tool_calls", []):
                            tc_id = tc.get("id")
                            if tc_id and tc_id not in seen_tcs:
                                seen_tcs.add(tc_id)
                                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                print(f"  [tool] {tc['name']}({args})", flush=True)
        print("\n")
        history = history + new_messages


if __name__ == "__main__":
    import main as _main

    _main.main()
