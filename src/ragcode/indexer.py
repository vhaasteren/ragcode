from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.llms.openai import OpenAI
from .config import Profile, ensure_dir
from .logging import info, warn
from .symbols import extract_python_symbols
from .embeddings import setup_embeddings_from_string

def _setup_llm():
    # Minimal: set an LLM for LlamaIndex (used for synthesis; embeddings configured via profile)
    Settings.llm = OpenAI(model="gpt-4o-mini")

def _setup_embeddings(profile: Profile):
    # Configure the embedding backend per profile.embed (openai|local|ollama)
    setup_embeddings_from_string(profile.embed)

def _tree_sitter_ok() -> bool:
    try:
        from tree_sitter_language_pack import get_parser  # noqa
        from tree_sitter import Parser  # noqa
        return True
    except Exception:
        return False

def _make_splitters(profile: Profile) -> Tuple[SentenceSplitter, Optional[CodeSplitter], bool]:
    sent = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    code = None
    use_code = False
    if _tree_sitter_ok():
        code = CodeSplitter(
            language="python",
            chunk_lines=profile.chunk_lines,
            chunk_lines_overlap=profile.chunk_overlap,
        )
        use_code = True
    else:
        warn("Tree-sitter not available; falling back to sentence splitting for code.")
    return sent, code, use_code

def _load_from_github(profile: Profile) -> List[Document]:
    token = os.environ.get("GITHUB_TOKEN")
    client = GithubClient(github_token=token, verbose=True)
    reader = GithubRepositoryReader(
        client,
        owner=profile.repo.split("/")[0],
        repo=profile.repo.split("/")[1],
        use_parser=False,
        verbose=True,
        concurrent_requests=2,
        filter_directories=(profile.include, GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions=(profile.ext, GithubRepositoryReader.FilterType.INCLUDE),
    )
    return reader.load_data(branch=profile.ref)

def _load_from_local(profile: Profile) -> List[Document]:
    base = Path(profile.path).resolve()
    docs: List[Document] = []
    for root, _, files in os.walk(base):
        rel = Path(root).relative_to(base)
        parts = list(rel.parts)
        if parts and parts[0] not in profile.include:
            continue
        for f in files:
            p = Path(root) / f
            if p.suffix not in profile.ext:
                continue
            if p.stat().st_size > profile.max_file_size_kb * 1024:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            d = Document(text=text, metadata={"file_path": str(p), "repo": "local", "file_ext": p.suffix})
            docs.append(d)
    return docs

def _commit_sha_hint(profile: Profile) -> str:
    # Best-effort; exact SHA from GitHubReader is non-trivial. Record ref + timestamp.
    return f"{profile.ref}@{datetime.utcnow().isoformat()}Z"

def build_index(profile: Profile) -> Dict[str, Any]:
    load_dotenv(".env")
    _setup_llm()
    _setup_embeddings(profile)

    persist_dir = Path(os.path.expanduser(profile.persist))
    ensure_dir(persist_dir)

    # Load documents
    info("Loading repository...")
    if profile.path:
        documents = _load_from_local(profile)
    elif profile.repo:
        documents = _load_from_github(profile)
    else:
        raise RuntimeError("Either 'path' (local) or 'repo' must be provided in profile/CLI.")

    # Annotate metadata
    for d in documents:
        if "file_path" not in d.metadata:
            d.metadata["file_path"] = d.id_
        d.metadata["file_ext"] = Path(d.metadata["file_path"]).suffix

    # Split
    info(f"Splitting {len(documents)} documents...")
    sent, code, use_code = _make_splitters(profile)
    nodes = []
    symbol_rows = []
    for d in documents:
        try:
            if use_code and d.metadata.get("file_ext") == ".py":
                nodes.extend(code.get_nodes_from_documents([d]))  # type: ignore
                # symbol index (python-only for now)
                symbol_rows.extend(extract_python_symbols(Path(d.metadata["file_path"]), d.text))
            else:
                nodes.extend(sent.get_nodes_from_documents([d]))
        except Exception as e:
            warn(f"Split error in {d.metadata.get('file_path')}: {e}")
            nodes.append(Document(text=d.text, metadata=d.metadata).as_related_node())

    # Build vector index and persist
    info(f"Indexing {len(nodes)} nodes...")
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(persist_dir))

    # Manifest
    manifest = {
        "repo": profile.repo or profile.path,
        "ref": profile.ref,
        "commit_sha_hint": _commit_sha_hint(profile),
        "persist_dir": str(persist_dir),
        "counts": {"documents": len(documents), "nodes": len(nodes), "symbols": len(symbol_rows)},
        "created": datetime.utcnow().isoformat() + "Z",
        "profile": profile.model_dump(),
    }
    (persist_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2))

    # Symbols sidecar
    if symbol_rows:
        (persist_dir / "symbols.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in symbol_rows))

    info(f"Done. Persisted index at {persist_dir}")
    return manifest

