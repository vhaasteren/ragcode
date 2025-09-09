from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # optional dependency


def _read_docstore(persist_dir: Path) -> Dict[str, Any]:
    """
    Read LlamaIndex simple_kvstore docstore. Supports common 0.13.x shapes:
    - {"docstore_data": { id: { "id_":..., "text":..., "metadata": {...} } }}
    - {"data": { id: { "id_":..., "text":..., "metadata": {...} } }}   (simple_kvstore)
    """
    docstore_path = persist_dir / "docstore.json"
    if not docstore_path.exists():
        return {}
    try:
        return json.loads(docstore_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_text_meta(obj: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Given a value from the kvstore, pull out (text, metadata).
    Handles both top-level and nested under "node".
    Returns ("", {}) if nothing usable is present.
    """
    if not isinstance(obj, dict):
        return "", {}

    # Common case: top-level text / metadata (TextNode serialization)
    text = obj.get("text")
    meta = obj.get("metadata")
    if isinstance(text, str) and text:
        return text, meta if isinstance(meta, dict) else {}

    # Nested under "node" (older/other shapes)
    node = obj.get("node")
    if isinstance(node, dict):
        text = node.get("text")
        meta = node.get("metadata")
        if isinstance(text, str) and text:
            return text, meta if isinstance(meta, dict) else {}

    # Some shapes store under "data" or other wrapper; try one more layer
    data = obj.get("data")
    if isinstance(data, dict):
        text = data.get("text")
        meta = data.get("metadata")
        if isinstance(text, str) and text:
            return text, meta if isinstance(meta, dict) else {}

    return "", {}


def _load_nodes_for_dump(persist_dir: Path) -> List[Dict[str, Any]]:
    """
    Produce a list of {"text": str, "metadata": dict} for all nodes in docstore.
    Works against LI 0.13.6 simple_kvstore.
    """
    raw = _read_docstore(persist_dir)
    container = None
    if "docstore_data" in raw and isinstance(raw["docstore_data"], dict):
        container = raw["docstore_data"]
    elif "data" in raw and isinstance(raw["data"], dict):
        container = raw["data"]
    else:
        return []

    out: List[Dict[str, Any]] = []
    for _, val in container.items():
        text, meta = _extract_text_meta(val)
        if text:
            if not isinstance(meta, dict):
                meta = {}
            out.append({"text": text, "metadata": meta})
    return out


def _bm25_topk(nodes: List[Dict[str, Any]], query: str, k: int) -> List[Dict[str, Any]]:
    """
    Score nodes with BM25 and return the top-k. Falls back to a simple heuristic if BM25 not present.
    """
    if not nodes:
        return []

    if BM25Okapi is not None:
        corpus = [n["text"].split() for n in nodes]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [nodes[i] for i in order]

    # Fallback: pick nodes with any query token, else first k
    terms = [t for t in query.lower().split() if len(t) > 2]
    picked = []
    for n in nodes:
        lt = n["text"].lower()
        if any(t in lt for t in terms):
            picked.append(n)
        if len(picked) >= k:
            break
    if not picked:
        picked = nodes[:k]
    return picked


def inspect_index(persist_dir: Path) -> str:
    manifest = persist_dir / "manifest.json"
    if not manifest.exists():
        return f"No manifest found at {persist_dir}."
    data = json.loads(manifest.read_text(encoding="utf-8"))
    lines = [
        f"Repo: {data.get('repo')}",
        f"Ref: {data.get('ref')}",
        f"Commit: {data.get('commit_sha_hint')}",
        f"Counts: {data.get('counts')}",
        f"Created: {data.get('created')}",
        f"Persist: {data.get('persist_dir')}",
    ]
    return "\n".join(lines)


def dump_snippets_md(persist_dir: Path, query: str, k: int, out_path: Path) -> Path:
    nodes = _load_nodes_for_dump(persist_dir)
    topk = _bm25_topk(nodes, query, k)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Context pack for: {query}\n\n")
        if topk:
            for i, n in enumerate(topk, 1):
                fp = n["metadata"].get("file_path", "?")
                f.write(f"## {i}. {fp}\n\n")
                f.write("```\n")
                f.write(n["text"][:5000])  # safety limit
                f.write("\n```\n\n")
            return out_path

        # ---- robust fallback: call the regular query pipeline so you at least get an answer+sources
        try:
            from .query import query_index  # local import to avoid cycles
            res = query_index(persist_dir, query, k=k, citations=True, hybrid=True, format_="md")
            f.write("_No nodes loaded from docstore; dumping synthesized answer instead._\n\n")
            f.write(res.get("content", ""))
        except Exception as e:
            f.write(f"_No matching snippets found; and query fallback failed: {e}_\n")
    return out_path


def dump_snippets_jsonl(persist_dir: Path, query: str, k: int, out_path: Path) -> Path:
    nodes = _load_nodes_for_dump(persist_dir)
    topk = _bm25_topk(nodes, query, k)
    if not topk:
        # fallback: single synthesized answer line
        try:
            from .query import query_index
            res = query_index(persist_dir, query, k=k, citations=True, hybrid=True, format_="jsonl")
            out_path.write_text(res.get("content", ""), encoding="utf-8")
            return out_path
        except Exception:
            pass

    lines = []
    for n in topk:
        lines.append({"role": "system", "content": n["text"], "metadata": n["metadata"]})
    out_path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return out_path

