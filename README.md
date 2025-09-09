# ragcode

Repo-aware RAG tooling to supercharge LLM-assisted coding. **ragcode** can index large codebases (PINT preset included), answer questions with citations, export snippet packs for Cursor/IDEs, serve a tiny API, and incrementally refresh its index as files change.

* **Embeddings:** OpenAI, local (HuggingFace), or Ollama.
* **Splitters:** sentence splitter plus tree-sitter-powered code splitter for Python.
* **Symbols:** lightweight Python symbol map (`symbols.jsonl`) for quick `explain`/`where`.
* **Index storage:** LlamaIndex JSON stores in `~/.ragcode/indexes/<profile>`.
* **Incremental indexing:** content-hash based; only changed files are re-embedded.
* **Optional reranker:** HuggingFace cross-encoder (e.g., `BAAI/bge-reranker-large`).

> ⚠️ **Web GUI is experimental:** the FastAPI UI is mainly for debugging and has not been battle-tested. Expect rough edges.

---

## Table of contents

* [Quick start](#quick-start)
* [Install](#install)
* [What each command is for (and when to use it)](#what-each-command-is-for-and-when-to-use-it)
* [Concepts](#concepts)
* [Profiles & configuration](#profiles--configuration)
* [Embedding backends](#embedding-backends)
* [CLI reference](#cli-reference)

  * [`inspect`](#inspect)
  * [`index`](#index)
  * [`query`](#query)
  * [`dump`](#dump)
  * [`explain` & `where`](#explain--where)
  * [`watch`](#watch)
  * [`serve` (API & experimental GUI)](#serve-api--experimental-gui)
* [Incremental indexing details](#incremental-indexing-details)
* [Outputs & on-disk layout](#outputs--on-disk-layout)
* [Known limitations](#known-limitations)
* [License](#license)

---

## Quick start

```bash
# 1) Install (editable)
pip install -e .

# 2) Set environment (OpenAI & optional GitHub)
export OPENAI_API_KEY=sk-...         # if using OpenAI embeddings/LLM
export GITHUB_TOKEN=ghp_...          # optional; improves GitHub rate limits

# 3) Index PINT (uses the packaged 'pint' profile)
ragcode index --profile pint

# 4) Ask a question with sources (natural-language answer)
ragcode query "simulate fake pulsar dataset ~1000 TOAs over 15yr" --k 8 --citations

# 5) Export raw code/text chunks for Cursor/IDE (no synthesis)
ragcode dump -q "ModelBuilder parse .par and residuals" --k 12 --out cursor_context.md

# 6) Start the tiny API (experimental GUI at /docs)
ragcode serve --host 127.0.0.1 --port 8008
```

### Incremental indexing

```bash
# Incremental (default): only changed files re-embedded
ragcode index --profile pint --incremental

# Force a full rebuild
ragcode index --profile pint --no-incremental
```

---

## Install

Python **3.10+** is required.

```bash
git clone <your fork or path> ragcode
cd ragcode
pip install -e .
```

The core dependencies include: Typer (CLI), Rich (TUI), FastAPI/Uvicorn (API), LlamaIndex (0.13.x), tree-sitter (optional for nicer code splitting), BM25 (rank-bm25), and sentence-transformers (for optional reranking).

---

## What each command is for (and **when** to use it)

Think of **ragcode** as two layers:

1. **Indexing layer** — prepares your repo for retrieval (`index`, `watch`, `inspect`).
2. **Retrieval layer** — uses that index for your workflow (`query`, `dump`, `explain`, `where`, `serve`).

### The big three you’ll use daily

| Command | What it returns                                              | When you want it                                                      | Why not the others?                                                               |
| ------- | ------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `query` | A **natural-language answer** + optional **citations**       | “Explain how to simulate TOAs” / “Which class builds a timing model?” | `dump` is raw snippets (no explanation), `explain/where` are symbol lookups only. |
| `dump`  | A **pack of top-k raw code/text chunks** (markdown or JSONL) | You need **verbatim** context to paste into Cursor/IDE or a prompt    | `query` might paraphrase or omit lines; `dump` guarantees raw snippets.           |
| `index` | (No user content) **Build/update** the index                 | Before your first `query`/`dump`, and whenever files change           | `query`/`dump` don’t rebuild; they rely on the index that `index` produces.       |

### Supporting commands

* `watch`: monitors a local path and re-indexes incrementally as you edit. Great during active development.
* `inspect`: sanity check that your index exists, how big it is, and where it lives.
* `explain`, `where`: quick symbol discovery for Python (functions/classes). Lightweight, fast.
* `serve`: tiny API with interactive docs; useful for hooking ragcode into tools or notebooks. (Experimental GUI.)

### Picking options that matter

* **`--k` (both `query`/`dump`)**: How many chunks to consider.

  * Small `k` (e.g., 5–8): faster, tighter answers.
  * Large `k` (e.g., 12–20): more coverage; better for `dump` packs.
* **`--citations` (`query`)**: Show where the answer came from. Turn **on** when you’ll paste the answer elsewhere or want traceability.
* **`--hybrid` (`query`)**: Shows BM25 **debug** candidates alongside vector search results. Useful to diagnose retrieval; not used for the final answer itself.
* **Reranker (profile-level)**: Improves ordering of retrieved chunks before synthesis. Use when your codebase is large and dense or class names collide.
* **Incremental indexing**: Default on. Keeps refresh fast and cheap by hashing files and only re-embedding changed ones.

**Rule of thumb**

* Use **`query`** to understand **how/why** and to get **links/snippets** to the evidence.
* Use **`dump`** when you need **exact code or doc fragments** to feed another tool (Cursor/IDE/LLM).
* Run **`index`** (or keep **`watch`** running) whenever the source moves.

---

## Concepts

* **Profile:** a named configuration that defines where your repo is (local path or `owner/repo@ref`), which files to index, embeddings backend, persistence path, etc. A PINT profile ships in `src/ragcode/profiles/pint.toml`.
* **Persisted index:** LlamaIndex stores plus ragcode sidecars (`manifest.json`, `filemap.json`, `symbols.jsonl`) under `~/.ragcode/indexes/<name>` by default.
* **Hybrid retrieval:** BM25 is used only for debug visibility (a candidate list); answers come from the vector index.
* **Reranker (optional):** a cross-encoder for better top-k ordering; opt-in via the profile.

---

## Profiles & configuration

Profiles are resolved in this order (later wins):

1. **Packaged profile**: `src/ragcode/profiles/<name>.toml` (e.g., `pint`)
2. **User profile**: `~/.ragcode/profiles/<name>.toml`
3. **Project-local**: `./.ragcode.toml` (use `--local-config` to point elsewhere)

Common profile keys (see `pint.toml` for an example):

```toml
name = "pint"
repo = "nanograv/PINT"        # or set 'path' to a local clone
ref = "master"
include = ["src", "docs", "examples", "tests"]
ext = [".py", ".md", ".rst", ".txt"]
persist = "~/.ragcode/indexes/pint"

# Embeddings backend (see next section)
embed = "openai:text-embedding-3-large"
# embed = "local:bge-base-en-v1.5"
# embed = "ollama:nomic-embed-text"

store = "llamaindex"
hybrid = true
reranker = "none"             # or "default" / "bge" / "BAAI/bge-reranker-large"
chunk_lines = 80
chunk_overlap = 20
symbol_chunks = true
parallel = 8
max_file_size_kb = 1024
```

You can override many of these per command with flags (e.g., `--path`, `--repo`, `--ref`, `--persist`).

---

## Embedding backends

Set `profile.embed` to one of:

* **OpenAI:** `openai:text-embedding-3-large` (default). Requires `OPENAI_API_KEY`.
* **Local (HuggingFace):** `local:<model-name>`, e.g. `local:bge-base-en-v1.5`.
* **Ollama:** `ollama:nomic-embed-text` (or another embedding model). Defaults to `http://localhost:11434`.

> The embedding model in use is written to `manifest.json` and later **reused** by `query`/`dump`, so dimensions match (no 1536↔3072 surprises).

---

## CLI reference

### `inspect`

Print a compact summary of the current index.

```bash
ragcode inspect --profile pint
ragcode inspect --local-config .ragcode.toml
```

**Options**

* `--profile TEXT` – profile name (e.g., `pint`)
* `--local-config PATH` – path to a project `.ragcode.toml`

---

### `index`

Build (or incrementally update) an index from a local path or GitHub repo.

```bash
# From the 'pint' profile as packaged
ragcode index --profile pint

# Override to a local checkout
ragcode index --profile pint --path /abs/path/to/PINT

# Force a full rebuild
ragcode index --profile pint --no-incremental
```

**Options**

* `--profile TEXT` – profile name (e.g., `pint`)
* `--local-config PATH` – project-local `.ragcode.toml`
* `--path PATH` – local repo path (overrides `profile.path`)
* `--repo TEXT` – `owner/repo` (overrides `profile.repo`)
* `--ref TEXT` – git ref/branch/tag
* `--persist PATH` – override persist directory
* `--incremental / --no-incremental` – default **on**; only changed files are re-embedded and node IDs updated accordingly

---

### `query`

Ask a question against the index, optionally with reranking and citations.

```bash
ragcode query "How do I construct a TimingModel?" --k 8 --citations
ragcode query "binary parameters" --format jsonl
```

**Arguments**

* `QUERY_STR` – your question

**Options**

* `--k INTEGER` – top-k nodes to retrieve (default 5)
* `--citations / --no-citations` – include a human-readable sources block (default on)
* `--hybrid / --no-hybrid` – show BM25 debug candidates (default on)
* `--format [md|jsonl|raw]` – output format (default `md`)
* `--profile TEXT`, `--persist PATH`, `--local-config PATH` – where to load the index from

**Why use `query`?**

* You want an **explanation** synthesized from the repo.
* You want to **skim** key files via the citations block, then drill in.
* You’re exploring unfamiliar parts of the codebase.

---

### `dump`

Export the top-k chunks (no synthesis) for pasting into Cursor/IDEs.

```bash
ragcode dump -q "residual calculation in ModelBuilder" --k 12 --out cursor_context.md
ragcode dump -q "TOA reading and flags" --format jsonl --out ctx.jsonl
```

**Options**

* `-q, --query TEXT` – topic to gather context for
* `--k INTEGER` – top-k chunks (default 12)
* `--out PATH` – output file (default `cursor_context.md`)
* `--format [md|jsonl]` – output format (default `md`)
* `--profile TEXT`, `--persist PATH`, `--local-config PATH`

**Why use `dump`?**

* You need **verbatim** code/doc fragments to feed another tool.
* You want a **portable pack** of context to share with a teammate.
* You prefer your IDE/LLM to do the reasoning, but with **grounded context**.

> `dump` uses the vector retriever directly (aligned to the index’s embedding model) and falls back to BM25 only if needed.

---

### `explain` & `where`

Lightweight symbol helpers using the sidecar `symbols.jsonl` (Python only for now).

```bash
ragcode explain TimingModel
ragcode where ModelBuilder
```

**Options (both commands)**

* `--profile TEXT`, `--persist PATH`, `--local-config PATH`

**Why use them?**

* Fast **“where is this defined?”** checks without a full query.
* Quick breadcrumbs before opening the file in your editor.

---

### `watch`

Watch a local path and rebuild the index when files change.

```bash
ragcode watch --path /abs/path/to/repo --profile pint
```

**Options**

* `--path PATH` – **required** local repo path to watch
* `--profile TEXT`, `--local-config PATH`
* `--debounce FLOAT` – seconds to debounce rebuilds (default 2.0)

**Why use it?**
Keep the index **always fresh** as you code; `query`/`dump` see new changes immediately.

---

### `serve` (API & experimental GUI)

Start a small FastAPI server. Interactive docs live at `http://HOST:PORT/docs`.

```bash
ragcode serve --host 127.0.0.1 --port 8008
```

**Options**

* `--host TEXT` – default `127.0.0.1`
* `--port INTEGER` – default `8008`
* `--reload / --no-reload` – uvicorn autoreload (off by default)

**Endpoints**

* `POST /query` – same fields as the CLI (`profile` or `persist_dir`, `query`, `k`, `citations`, `hybrid`, `format`)
* `POST /explain` – `symbol` plus `profile`/`persist_dir`
* `POST /where` – `symbol` plus `profile`/`persist_dir`
* `POST /inspect` – returns `manifest.json` text
* `GET /healthz` – returns `ok`

> ⚠️ **Experimental:** the API/GUI is primarily a debug aid; stability and auth are minimal.

---

## Incremental indexing details

* ragcode maintains a `filemap.json` that maps **canonical file paths** to `{ hash, node_ids[] }`.
* During `index --incremental`, ragcode:

  1. Loads current files (from local path or GitHub) and computes SHA-256 hashes.
  2. Compares with `filemap.json` to find **added / modified / removed** files.
  3. Deletes nodes for removed/modified files and inserts nodes for added/modified files.
  4. Updates `filemap.json`, `manifest.json`, and `symbols.jsonl`.
* If the embedding/backend or **source** changes (e.g., from GitHub to local path), ragcode will fall back to a **full rebuild** for safety.

---

## Outputs & on-disk layout

Under `~/.ragcode/indexes/<profile>` (or your `persist` path) you’ll find:

* **`docstore.json`, `index_store.json`, …** – LlamaIndex storage.
* **`manifest.json`** – build summary (counts, timestamps, profile snapshot, embed spec).
* **`filemap.json`** – per-file `{hash, node_ids[]}` map for incremental updates.
* **`symbols.jsonl`** – Python symbol rows (function/class with file/line span).

These files allow `query`/`dump` to **reuse** the exact embedding model used at index time and keep updates cheap.

---

## Known limitations

* **Language coverage:** code-aware splitting + symbol extraction target Python right now. Other languages fall back to sentence splitting; symbol maps are not produced.
* **Reranker model size:** cross-encoders (e.g., `bge-reranker-large`) are heavy and may use significant RAM/VRAM.
* **BM25 is debug-only:** we show its top candidates for visibility; answers/snippets come from the vector index.
* **File renames:** counted as remove+add (nodes are regenerated).
* **Web GUI:** experimental; no auth, limited features.

---

## License

MIT © Rutger + ChatGPT

