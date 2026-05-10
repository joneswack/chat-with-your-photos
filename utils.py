import base64
import io
import os
import re
import subprocess
import sys
import termios
import tty
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

REGISTRY = Path.home() / ".cwyp" / "folders.txt"


def encode_jpeg_b64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def join_nonempty(*parts: Any, sep: str = ", ") -> str:
    """Join string parts, skipping anything that isn't a non-blank string."""
    return sep.join(p for p in parts if isinstance(p, str) and p.strip())


def pick(prompt: str, options: list[str]) -> str:
    idx = 0
    n = len(options)

    def render(first: bool) -> None:
        if not first:
            sys.stdout.write(f"\x1b[{n + 1}A")
        sys.stdout.write(f"\r{prompt}\r\n")
        for i, opt in enumerate(options):
            marker = "> " if i == idx else "  "
            sys.stdout.write(f"  {marker}{opt}\r\n")
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        render(True)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x04":
                raise EOFError
            if ch == "\x1b":
                rest = sys.stdin.read(2)
                if rest == "[A":
                    idx = (idx - 1) % n
                elif rest == "[B":
                    idx = (idx + 1) % n
            render(False)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    sys.stdout.write("\r\n")
    sys.stdout.flush()
    return options[idx]


def anthropic_models() -> list[str]:
    """Return current Claude model IDs from the Anthropic API, or [] on failure."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    try:
        import anthropic

        client = anthropic.Anthropic()
        return [
            m.id
            for m in client.models.list()
            if m.id.startswith("claude-")
            and not re.search(r"-\d{8}$", m.id)  # exclude date-versioned IDs
            and re.search(r"-[4-9]", m.id)  # current generation only (4+)
        ]
    except Exception as e:
        print(f"Warning: could not fetch Anthropic models: {e}", file=sys.stderr)
        return []


def openai_models() -> list[str]:
    """Return models from the two most recent GPT families, newest-first from the API."""
    if not os.environ.get("OPENAI_API_KEY"):
        return []

    # Exclude fine-tunes, instruct, realtime, audio, and date-pinned variants
    _EXCLUDE = re.compile(r"(instruct|realtime|audio|-\d{4}-\d{2}-\d{2}$|-\d{8}$|:)")

    try:
        import openai

        client = openai.OpenAI()
        # OpenAI returns models newest-first; keep only plain gpt- models
        gpt_ids = [
            m.id
            for m in client.models.list()
            if m.id.startswith("gpt-") and not _EXCLUDE.search(m.id)
        ]

        # Group by family (e.g. "gpt-4.1-mini" → family "4.1")
        family_models: dict[str, list[str]] = {}
        for model_id in gpt_ids:
            m = re.match(r"gpt-([^-]+)", model_id)
            family = m.group(1) if m else model_id
            family_models.setdefault(family, []).append(model_id)

        def _version_key(family: str) -> tuple[int, float]:
            # "5.5" → (5, 5.0), "5" → (5, 0.0), "4.1" → (4, 1.0), "4o" → (4, 0.0)
            nums = re.match(r"(\d+)(?:\.(\d+))?", family)
            if not nums:
                return (0, 0.0)
            return (int(nums.group(1)), float(nums.group(2) or 0))

        top_families = sorted(family_models, key=_version_key, reverse=True)[:2]

        result = []
        for family in top_families:
            result.extend(sorted(family_models[family]))
        return result
    except Exception as e:
        print(f"Warning: could not fetch OpenAI models: {e}", file=sys.stderr)
        return []


def ollama_models() -> list[str]:
    """Return models available in the local Ollama instance, or [] if not installed."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        # First line is the header (NAME  ID  SIZE  MODIFIED), skip it
        return [line.split()[0] for line in lines[1:] if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        return []


def register_folder(folder: Path) -> None:
    """Add folder to the global registry so agent.py can discover it."""
    REGISTRY.parent.mkdir(exist_ok=True)
    existing: set[str] = set()
    if REGISTRY.exists():
        existing = {ln.strip() for ln in REGISTRY.read_text().splitlines() if ln.strip()}
    if str(folder) not in existing:
        with REGISTRY.open("a") as f:
            f.write(str(folder) + "\n")


def indexed_folders() -> list[Path]:
    """Return all folders that have been indexed by setup.py and still exist."""
    if not REGISTRY.exists():
        return []
    folders = []
    for line in REGISTRY.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.is_dir() and (p / ".cwyp" / "index.csv").exists():
            folders.append(p)
    return folders
