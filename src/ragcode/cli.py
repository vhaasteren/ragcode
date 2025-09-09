from __future__ import annotations
import sys
from pathlib import Path
import typer
from rich import print as rprint
from .config import load_profile, ensure_dir, Profile
from .logging import info, warn, err
from .indexer import build_index
from .query import query_index
from .explain_where import explain_symbol, where_symbol
from .inspect_dump import inspect_index, dump_snippets_md, dump_snippets_jsonl
from .server import app as fastapi_app
import uvicorn
import time

app = typer.Typer(add_completion=False, help="Repo-aware RAG CLI for LLM-assisted coding.")

def _profile(profile: str | None, local_config: str | None) -> Profile:
    return load_profile(profile, Path(local_config) if local_config else Path(".ragcode.toml"))

@app.command()
def inspect(
    profile: str = typer.Option(None, help="Profile name (e.g., pint)"),
    local_config: str = typer.Option(None, help="Project-local .ragcode.toml"),
):
    p = _profile(profile, local_config)
    persist = Path(p.persist).expanduser()
    rprint(inspect_index(persist))

@app.command()
def index(
    profile: str = typer.Option(None, help="Profile name (e.g., pint)"),
    local_config: str = typer.Option(None, help="Project-local .ragcode.toml"),
    path: str = typer.Option(None, help="Local repo path (overrides profile.path)"),
    repo: str = typer.Option(None, help="GitHub owner/repo (overrides profile.repo)"),
    ref: str = typer.Option(None, help="Git ref/branch/tag (overrides profile.ref)"),
    persist: str = typer.Option(None, help="Persist dir (overrides profile.persist)"),
):
    p = _profile(profile, local_config)
    if path: p.path = path
    if repo: p.repo = repo
    if ref: p.ref = ref
    if persist: p.persist = persist
    manifest = build_index(p)
    rprint(manifest)

@app.command()
def query(
    query_str: str = typer.Argument(..., help="Your question"),
    k: int = typer.Option(5, "--k"),
    citations: bool = typer.Option(True, "--citations/--no-citations"),
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid"),
    format: str = typer.Option("md", help="md|jsonl|raw"),
    profile: str = typer.Option(None, help="Profile name"),
    persist: str = typer.Option(None, help="Persist dir override"),
    local_config: str = typer.Option(None, help="Project-local .ragcode.toml"),
):
    p = _profile(profile, local_config)
    persist_dir = Path(persist or p.persist).expanduser()
    res = query_index(persist_dir, query_str, k, citations, hybrid, format)
    if res["format"] in ("md", "raw"):
        rprint(res["content"])
    elif res["format"] == "jsonl":
        print(res["content"])

@app.command()
def explain(
    symbol: str = typer.Argument(..., help="Function/Class name"),
    profile: str = typer.Option(None),
    persist: str = typer.Option(None),
    local_config: str = typer.Option(None),
):
    p = _profile(profile, local_config)
    persist_dir = Path(persist or p.persist).expanduser()
    rprint(explain_symbol(persist_dir, symbol))

@app.command()
def where(
    symbol: str = typer.Argument(..., help="Function/Class name"),
    profile: str = typer.Option(None),
    persist: str = typer.Option(None),
    local_config: str = typer.Option(None),
):
    p = _profile(profile, local_config)
    persist_dir = Path(persist or p.persist).expanduser()
    rprint(where_symbol(persist_dir, symbol))

@app.command()
def dump(
    query: str = typer.Option(..., "--query", "-q", help="Topic to gather context for"),
    k: int = typer.Option(12, "--k"),
    out: str = typer.Option("cursor_context.md", "--out"),
    format: str = typer.Option("md", help="md|jsonl"),
    profile: str = typer.Option(None),
    persist: str = typer.Option(None),
    local_config: str = typer.Option(None),
):
    p = _profile(profile, local_config)
    persist_dir = Path(persist or p.persist).expanduser()
    out_path = Path(out)
    if format == "md":
        dump_snippets_md(persist_dir, query, k, out_path)
    else:
        dump_snippets_jsonl(persist_dir, query, k, out_path)
    rprint(f"Written {format.upper()} to {out_path}")

@app.command()
def watch(
    path: str = typer.Option(..., help="Local repo to watch"),
    profile: str = typer.Option(None),
    local_config: str = typer.Option(None),
    debounce: float = typer.Option(2.0, help="Seconds to debounce rebuilds"),
):
    """
    Watch a local path and re-index on file changes (incremental in future; full rebuild now).
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except Exception:
        err("watchdog not installed. `pip install watchdog`")
        raise typer.Exit(1)

    p = _profile(profile, local_config)
    p.path = path

    class Handler(FileSystemEventHandler):
        def __init__(self):
            self._last = 0.0
        def on_any_event(self, event):
            nonlocal p
            now = time.time()
            if now - self._last < debounce:  # debounce
                return
            self._last = now
            info("Change detected. Re-indexing...")
            build_index(p)

    obs = Observer()
    obs.schedule(Handler(), path, recursive=True)
    obs.start()
    info("Watching. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8008),
    reload: bool = typer.Option(False),
):
    uvicorn.run("ragcode.server:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    app()

