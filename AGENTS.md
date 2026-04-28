# Chat with Your Photos

## Goal

Answer arbitrary natural-language queries about your collection of photos.

Example queries:
- "What places did I visit?"
- "How many photos did I take in the last 7 days?"
- "What animals are in the photos?"
- "Create a trajectory map of my Mallorca holiday!"
- "Create a postcard with the nicest pictures from the trip to Africa!"

The app operates on a user-specified folder (and all subfolders) of photos. It pre-indexes them so the agent can answer queries without loading every image at query time.

---

## Repository Structure

```
main.py           — Top-level entry point: welcome screen, folder picker, dispatches to agent or setup
setup.py          — Indexes a photo collection into a CSV file
agent.py          — LangGraph ReAct agent that answers queries using the CSV index; receives a folder path from main.py
utils.py          — Shared utilities used by all modules
pyproject.toml    — Dependencies (managed with uv)
.env              — API keys (ANTHROPIC_API_KEY); not committed
```

`utils.py` provides: `pick` (interactive arrow-key menu), `anthropic_models` / `openai_models` / `ollama_models` (model discovery), `REGISTRY` path constant, `register_folder` (called by setup.py), and `indexed_folders` (called by main.py).

All other files not mentioned here are still experimental and yet to be integrated.

The index lives at `{IMAGES_DIR}/.cwyp/index.csv` — a hidden subfolder inside the user's photo directory, so the index travels with the photos.

A global registry at `~/.cwyp/folders.txt` lists every folder that has been indexed (one absolute path per line). `setup.py` appends to this file after a successful run. `main.py` reads it at startup to show how many folders are indexed and offer a collection picker — no manual path entry required.

---

## Index Format

The CSV produced by `setup.py` and consumed by `agent.py` has these columns:

| Column               | Type / Example                    | Notes                                             |
|----------------------|-----------------------------------|---------------------------------------------------|
| `relative_file_path` | `2024/Mallorca/IMG_1234.HEIC`     | Relative path from `IMAGES_DIR`                   |
| `label`              | `African elephant, Loxodonta`     | Class label from ImageNet21k or LLM (≤ 3 words for LLM) |
| `description`        | `A herd of elephants at sunset.`  | One-sentence description; LLM only, empty for timm |
| `timestamp`          | `2024:07:14 09:31:00`             | EXIF DateTimeOriginal; may be empty               |
| `lat`                | `39.5696`                         | GPS latitude (decimal degrees); empty if no GPS   |
| `lon`                | `2.6502`                          | GPS longitude (decimal degrees); empty if no GPS  |
| `country`            | `Spain`                           | Reverse-geocoded from GPS; empty if no GPS        |
| `region`             | `Balearic Islands`                | Admin1 region                                     |
| `city`               | `Palma`                           | Nearest city                                      |

`agent.py` parses `timestamp` with `pd.to_datetime(..., format="%Y:%m:%d %H:%M:%S", errors="coerce")`.

---

## Tech Stack

| Layer            | Library / Model                                              |
|------------------|--------------------------------------------------------------|
| Indexing vision  | `timm` — `vit_base_patch16_224.augreg_in21k` (ImageNet21k, fast, label only) or any Anthropic / OpenAI / Ollama vision model (label + description) |
| GPS decoding     | EXIF IFD 34853 → `reverse_geocoder` → `pycountry`           |
| Image formats    | `.jpg`, `.jpeg`, `.png`, `.heic`                             |
| Agent framework  | LangChain + LangGraph (`create_agent`)                       |
| Agent LLM        | Any Anthropic Claude (4+), OpenAI GPT (top 2 families), or Ollama model — picked interactively at startup |
| Shared utilities | `utils.py` — model discovery, interactive picker, registry helpers |
| Data wrangling   | `pandas`                                                     |
| Env/secrets      | `python-dotenv` (reads `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` from `.env`) |

---

## How to Run

```bash
# Install dependencies
uv sync

# Start the app (folder selection, indexing, and chat all flow from here)
uv run main.py
```

Set `ANTHROPIC_API_KEY` in `.env` before running the agent if using Anthropic models.
Set `OPENAI_API_KEY` in `.env` before running the agent if using OpenAI models.
Make sure ollama is installed if using ollama models.

---

## Conventions

- **No interactive UI** — the agent is a terminal REPL; keep it that way unless asked.
- **Index allows for quick navigation** — `agent.py` initially operates on the index via pandas functions to be quick. It only investigates images directly when needed.
- **Compact index** — each row must be short enough that the agent can reason over many rows in one context window. Avoid multi-sentence free-text columns.
- **Python 3.12+**, managed with `uv`. Do not use `pip` or `conda`.
- **Agent construction** — use `from langchain.agents import create_agent`, not `from langgraph.prebuilt import create_react_agent` (deprecated since LangGraph v1.0).
