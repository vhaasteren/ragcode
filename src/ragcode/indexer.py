from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, DefaultDict
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter, CodeSplitter
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.llms.openai import OpenAI
from .config import Profile, ensure_dir
from .logging import info, warn
from .symbols import extract_python_symbols
from .embeddings import setup_embeddings_from_string

# ---------- helpers ----------

def _setup_llm():
    Settings.llm = OpenAI(model="gpt-4o-mini")

def _setup_embeddings(profile: Profile):
    setup_embeddings_from_string(profile.embed)

def _tree_sitter_ok() -> bool:
    try:
        from tree_sitter_language_pack import get_parser  # noqa: F401
        from tree_sitter import Parser  # noqa: F401
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

def _canon_for_local(base: Path, p: Path) -> str:
    try:
        rel = p.resolve().relative_to(base.resolve())
    except Exception:
        rel = p.name
    return str(rel).replace("\\", "/")

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
    docs = reader.load_data(branch=profile.ref)
    for d in docs:
        fp = d.id_
        d.metadata.setdefault("file_path", fp)
        d.metadata["canonical_path"] = str(fp).replace("\\", "/")
        d.metadata["file_ext"] = Path(fp).suffix
        d.metadata.setdefault("repo", profile.repo or "github")
    return docs

def _load_from_local(profile: Profile) -> List[Document]:
    base = Path(profile.path).resolve()
    docs: List[Document] = []
    for root, _, files in os.walk(base):
        rel_top = Path(root).resolve().relative_to(base)
        parts = list(rel_top.parts)
        if parts and parts[0] not in profile.include:
            continue
        for f in files:
            p = Path(root) / f
            if p.suffix not in profile.ext:
                continue
            if p.stat().st_size > profile.max_file_size_kb * 1024:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            canon = _canon_for_local(base, p)
            d = Document(
                text=text,
                metadata={
                    "file_path": str(p.resolve()),
                    "canonical_path": canon,
                    "repo": "local",
                    "file_ext": p.suffix,
                },
            )
            docs.append(d)
    return docs

def _commit_sha_hint(profile: Profile) -> str:
    return f"{profile.ref}@{datetime.utcnow().isoformat()}Z"

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _load_filemap(persist_dir: Path) -> Dict[str, Any]:
    fp = persist_dir / "filemap.json"
    if not fp.exists():
        return {"version": 1, "source": None, "files": {}}
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
        raw.setdefault("version", 1)
        raw.setdefault("source", None)
        raw.setdefault("files", {})
        return raw
    except Exception:
        return {"version": 1, "source": None, "files": {}}

def _save_filemap(persist_dir: Path, fm: Dict[str, Any]) -> None:
    (persist_dir / "filemap.json").write_text(json.dumps(fm, indent=2), encoding="utf-8")

def _get_node_id(node) -> str:
    return getattr(node, "node_id", None) or getattr(node, "id_", None)

def _load_existing_index(persist_dir: Path) -> Optional[VectorStoreIndex]:
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        return load_index_from_storage(storage_context)
    except Exception:
        return None

def _source_fingerprint(profile: Profile) -> str:
    if profile.path:
        return f"local:{Path(profile.path).resolve()}"
    if profile.repo:
        return f"github:{profile.repo}@{profile.ref}"
    return "unknown"

def _read_docstore_nodes(persist_dir: Path) -> List[Dict[str, Any]]:
    ds = persist_dir / "docstore.json"
    if not ds.exists():
        return []
    try:
        raw = json.loads(ds.read_text(encoding="utf-8"))
    except Exception:
        return []
    container = None
    if "docstore_data" in raw and isinstance(raw["docstore_data"], dict):
        container = raw["docstore_data"]
    elif "data" in raw and isinstance(raw["data"], dict):
        container = raw["data"]
    else:
        return []
    out = []
    for _, node in container.items():
        text = node.get("text") or node.get("node", {}).get("text") or ""
        meta = node.get("metadata", {}) or node.get("node", {}).get("metadata", {})
        nid = node.get("id_", None)
        cpath = (meta.get("canonical_path") or meta.get("file_path") or "").replace("\\", "/")
        out.append({"id": nid, "text": text, "metadata": meta, "canonical_path": cpath})
    return out

def _bootstrap_filemap_if_missing(persist_dir: Path, filemap: Dict[str, Any], current_hashes_by_canon: Dict[str, str]) -> Dict[str, Any]:
    if filemap.get("files"):
        return filemap
    nodes = _read_docstore_nodes(persist_dir)
    if not nodes:
        return filemap
    buckets: DefaultDict[str, List[str]] = defaultdict(list)
    for n in nodes:
        cp = n.get("canonical_path") or ""
        nid = n.get("id")
        if cp and nid:
            buckets[cp].append(nid)
    if not buckets:
        return filemap
    boot = {"version": 1, "source": filemap.get("source"), "files": {}}
    for cp, nids in buckets.items():
        h = current_hashes_by_canon.get(cp)
        boot["files"][cp] = {"hash": h or "", "node_ids": nids}
    _save_filemap(persist_dir, boot)
    return boot

def _force_node_meta(node, dmeta: Dict[str, Any]) -> None:
    """
    Ensure nodes carry canonical_path and file_ext copied from the parent document metadata.
    Some splitters drop custom fields; enforce them here.
    """
    nmeta = getattr(node, "metadata", {}) or {}
    if "canonical_path" not in nmeta and "canonical_path" in dmeta:
        nmeta["canonical_path"] = dmeta["canonical_path"]
    if "file_ext" not in nmeta and "file_ext" in dmeta:
        nmeta["file_ext"] = dmeta["file_ext"]
    # write back
    node.metadata = nmeta

# ---------- main build ----------

def build_index(profile: Profile, incremental: bool = True) -> Dict[str, Any]:
    load_dotenv(".env")
    _setup_llm()
    _setup_embeddings(profile)

    persist_dir = Path(os.path.expanduser(profile.persist))
    ensure_dir(persist_dir)

    # Load documents
    info("Loading repository...")
    if profile.path:
        documents = _load_from_local(profile)
        source_mode = "local"
        base_for_local = Path(profile.path).resolve()
    elif profile.repo:
        documents = _load_from_github(profile)
        source_mode = "github"
        base_for_local = None
    else:
        raise RuntimeError("Either 'path' (local) or 'repo' must be provided in profile/CLI.")

    # Compute per-file current hashes keyed by canonical path
    cur_files: Dict[str, Dict[str, Any]] = {}
    current_hashes_by_canon: Dict[str, str] = {}
    for d in documents:
        meta = d.metadata
        if "canonical_path" not in meta:
            fp = meta.get("file_path") or getattr(d, "id_", "")
            if source_mode == "local" and base_for_local is not None:
                meta["canonical_path"] = _canon_for_local(base_for_local, Path(fp))
            else:
                meta["canonical_path"] = str(fp).replace("\\", "/")
        meta["file_ext"] = meta.get("file_ext") or Path(meta["canonical_path"]).suffix
        cpath = meta["canonical_path"]
        fh = _sha256_text(d.text)
        cur_files[cpath] = {"hash": fh, "doc": d}
        current_hashes_by_canon[cpath] = fh

    # Load existing filemap and handle source changes
    filemap = _load_filemap(persist_dir)
    desired_source = _source_fingerprint(profile)
    if filemap.get("source") and filemap["source"] != desired_source:
        warn(f"Source changed from '{filemap['source']}' to '{desired_source}'. For safety, doing full rebuild.")
        incremental = False

    # Bootstrap filemap if needed (existing docstore but no filemap)
    if incremental and (persist_dir / "docstore.json").exists() and not filemap.get("files"):
        filemap = _bootstrap_filemap_if_missing(persist_dir, filemap, current_hashes_by_canon)

    prev_files: Dict[str, Any] = filemap.get("files", {})

    # Determine changes
    added_paths: Set[str] = set()
    modified_paths: Set[str] = set()
    removed_paths: Set[str] = set()

    if incremental and (persist_dir / "docstore.json").exists():
        prev_paths = set(prev_files.keys())
        curr_paths = set(cur_files.keys())
        added_paths = curr_paths - prev_paths
        removed_paths = prev_paths - curr_paths
        for pth in (curr_paths & prev_paths):
            if cur_files[pth]["hash"] != prev_files[pth].get("hash"):
                modified_paths.add(pth)
        if not (added_paths or modified_paths or removed_paths):
            info("No changes detected; index is up to date.")
            # counts from filemap (reliable) + symbols sidecar
            nodes_count = sum(len(v.get("node_ids", [])) for v in prev_files.values())
            symbols_count = 0
            sp = persist_dir / "symbols.jsonl"
            if sp.exists():
                symbols_count = sum(1 for _ in sp.open())
            manifest = {
                "repo": profile.repo or profile.path,
                "ref": profile.ref,
                "commit_sha_hint": _commit_sha_hint(profile),
                "persist_dir": str(persist_dir),
                "counts": {"documents": len(documents), "nodes": nodes_count, "symbols": symbols_count},
                "created": datetime.utcnow().isoformat() + "Z",
                "profile": profile.model_dump(),
            }
            (persist_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            # ensure source + hashes present
            filemap["source"] = desired_source
            for cp, meta in filemap.get("files", {}).items():
                if not meta.get("hash") and cp in current_hashes_by_canon:
                    meta["hash"] = current_hashes_by_canon[cp]
            _save_filemap(persist_dir, filemap)
            return manifest
    else:
        added_paths = set(cur_files.keys())
        modified_paths = set()
        removed_paths = set(prev_files.keys()) if prev_files else set()

    # Load or create index
    index = None
    if (persist_dir / "docstore.json").exists() and incremental:
        index = _load_existing_index(persist_dir)
        if index is None:
            warn("Existing storage found but failed to load; falling back to full rebuild.")
    if index is None:
        # -------- fresh build --------
        info("Creating new index storage...")
        sent, code, use_code = _make_splitters(profile)
        all_nodes = []
        symbol_rows = []
        for d in documents:
            try:
                if use_code and d.metadata.get("file_ext") == ".py":
                    nodes = code.get_nodes_from_documents([d])  # type: ignore
                    symbol_rows.extend(extract_python_symbols(Path(d.metadata["file_path"]), d.text))
                else:
                    nodes = sent.get_nodes_from_documents([d])
            except Exception as e:
                warn(f"Split error in {d.metadata.get('file_path')}: {e}")
                nodes = [Document(text=d.text, metadata=d.metadata).as_related_node()]

            # **Ensure canonical metadata is present on every node**
            for n in nodes:
                _force_node_meta(n, d.metadata)

            all_nodes.extend(nodes)

        index = VectorStoreIndex(all_nodes)
        index.storage_context.persist(persist_dir=str(persist_dir))

        # Build fresh filemap using canonical paths
        new_map: Dict[str, Any] = {"version": 1, "source": desired_source, "files": {}}
        for n in all_nodes:
            meta = getattr(n, "metadata", {}) or {}
            cp = meta.get("canonical_path") or meta.get("file_path")
            if not cp:
                continue
            nid = _get_node_id(n)
            entry = new_map["files"].setdefault(cp, {"hash": cur_files.get(cp, {}).get("hash", ""), "node_ids": []})
            if nid:
                entry["node_ids"].append(nid)
        _save_filemap(persist_dir, new_map)

        # Symbols sidecar
        if symbol_rows:
            (persist_dir / "symbols.jsonl").write_text("\n".join(json.dumps(r) for r in symbol_rows), encoding="utf-8")

        # counts (robust)
        nodes_count = len(all_nodes)
        symbols_count = len(symbol_rows)

    else:
        # -------- incremental update --------
        info(f"Incremental update: +{len(added_paths)} / ~{len(modified_paths)} / -{len(removed_paths)}")
        # helpful debug sample
        if added_paths:
            sample = list(sorted(added_paths))[:5]
            warn(f"Added sample (first 5): {sample}")

        # Delete nodes for removed + modified files
        to_delete_ids: List[str] = []
        for pth in removed_paths | modified_paths:
            ids = prev_files.get(pth, {}).get("node_ids", [])
            to_delete_ids.extend(ids)
        if to_delete_ids:
            try:
                index.delete_nodes(node_ids=to_delete_ids)
            except Exception as e:
                warn(f"delete_nodes failed ({len(to_delete_ids)} ids): {e}")

        # Split & insert nodes for added + modified files
        sent, code, use_code = _make_splitters(profile)
        new_symbol_rows: List[Dict[str, Any]] = []
        updated_entries: Dict[str, Any] = {}

        for cp in sorted(list(added_paths | modified_paths)):
            d = cur_files[cp]["doc"]
            try:
                if use_code and d.metadata.get("file_ext") == ".py":
                    nodes = code.get_nodes_from_documents([d])  # type: ignore
                    new_symbol_rows.extend(extract_python_symbols(Path(d.metadata["file_path"]), d.text))
                else:
                    nodes = sent.get_nodes_from_documents([d])
            except Exception as e:
                warn(f"Split error in {cp}: {e}")
                nodes = [Document(text=d.text, metadata=d.metadata).as_related_node()]

            # **Ensure canonical metadata is present on every node**
            for n in nodes:
                _force_node_meta(n, d.metadata)

            try:
                index.insert_nodes(nodes)
            except Exception as e:
                warn(f"insert_nodes failed for {cp}: {e}")

            entry = {"hash": cur_files[cp]["hash"], "node_ids": []}
            for n in nodes:
                nid = _get_node_id(n)
                if nid:
                    entry["node_ids"].append(nid)
            updated_entries[cp] = entry

        # Update filemap
        new_map = {"version": 1, "source": desired_source, "files": {}}
        for cp, meta in prev_files.items():
            if cp in removed_paths or cp in modified_paths:
                continue
            new_map["files"][cp] = meta
        for cp, meta in updated_entries.items():
            new_map["files"][cp] = meta
        _save_filemap(persist_dir, new_map)

        # Update symbols sidecar
        symp = persist_dir / "symbols.jsonl"
        if symp.exists():
            existing = [json.loads(line) for line in symp.read_text(encoding="utf-8").splitlines() if line.strip()]
            existing = [
                r for r in existing
                if (r.get("canonical_path", r.get("file_path")) not in modified_paths)
                and (r.get("canonical_path", r.get("file_path")) not in removed_paths)
            ]
        else:
            existing = []
        if new_symbol_rows:
            existing.extend(new_symbol_rows)
        if existing:
            symp.write_text("\n".join(json.dumps(r) for r in existing), encoding="utf-8")
        elif symp.exists():
            symp.unlink(missing_ok=True)

        # Persist storage changes
        index.storage_context.persist(persist_dir=str(persist_dir))

        # counts from filemap
        nodes_count = sum(len(v.get("node_ids", [])) for v in new_map["files"].values())
        symbols_count = 0
        sp = persist_dir / "symbols.jsonl"
        if sp.exists():
            symbols_count = sum(1 for _ in sp.open())

    # Final manifest
    manifest = {
        "repo": profile.repo or profile.path,
        "ref": profile.ref,
        "commit_sha_hint": _commit_sha_hint(profile),
        "persist_dir": str(persist_dir),
        "counts": {"documents": len(documents), "nodes": nodes_count, "symbols": symbols_count},
        "created": datetime.utcnow().isoformat() + "Z",
        "profile": profile.model_dump(),
    }
    (persist_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    info(f"Done. Persisted index at {persist_dir}")
    return manifest

