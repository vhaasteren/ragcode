from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.llms.openai import OpenAI
from rank_bm25 import BM25Okapi
from .logging import info
from .embeddings import setup_embeddings_from_string
import json

# Optional: sentence-transformers reranker (if enabled in profile)
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank  # available in LI 0.13.x
except Exception:
    SentenceTransformerRerank = None  # type: ignore


def _setup_llm():
    Settings.llm = OpenAI(model="gpt-4o-mini")


def _load_index(persist_dir: Path):
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    return index


def _load_manifest(persist_dir: Path) -> Dict[str, Any]:
    mf = persist_dir / "manifest.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_manifest_embed(persist_dir: Path) -> Optional[str]:
    data = _load_manifest(persist_dir)
    prof = data.get("profile") or {}
    return prof.get("embed")


def _load_manifest_reranker(persist_dir: Path) -> Optional[str]:
    data = _load_manifest(persist_dir)
    prof = data.get("profile") or {}
    return (prof.get("reranker") or "none").lower()


def _load_nodes_texts(persist_dir: Path) -> List[Dict[str, Any]]:
    # LlamaIndex persists nodes in docstore.json; we'll read it to build BM25 corpus
    docstore_path = persist_dir / "docstore.json"
    if not docstore_path.exists():
        return []
    data = json.loads(docstore_path.read_text(encoding="utf-8"))
    # Handle common shapes ("docstore_data" or "data")
    container = None
    if "docstore_data" in data and isinstance(data["docstore_data"], dict):
        container = data["docstore_data"]
    elif "data" in data and isinstance(data["data"], dict):
        container = data["data"]
    else:
        return []
    result = []
    for _, node in container.items():
        text = (
            node.get("text")
            or node.get("node", {}).get("text")
            or ""
        )
        meta = node.get("metadata", {}) or node.get("node", {}).get("metadata", {})
        result.append({"text": text, "metadata": meta, "ref_id": node.get("id_", "")})
    return result


def _bm25_candidates(query: str, nodes: List[Dict[str, Any]], top_k: int) -> List[int]:
    if not nodes:
        return []
    tokenized_corpus = [n["text"].split() for n in nodes]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return idxs


def _maybe_make_reranker(reranker_name: str | None):
    """
    Construct a reranker postprocessor if requested and available.
    Supports simple 'bge-reranker-large' (or any HF cross-encoder name) via SentenceTransformerRerank.
    """
    if not reranker_name or reranker_name in {"", "none"}:
        return None
    if SentenceTransformerRerank is None:
        return None
    # Sensible default if user sets 'bge' or 'default'
    if reranker_name in {"default", "bge"}:
        model = "BAAI/bge-reranker-large"
    else:
        model = reranker_name
    try:
        return SentenceTransformerRerank(model=model, top_n=10)
    except Exception:
        return None


def _format_sources_md(response, top_n: int = 8) -> str:
    """
    Build a human-friendly citations block from response.source_nodes,
    showing file_path, score, and a short snippet.
    """
    if not hasattr(response, "source_nodes") or not response.source_nodes:
        return ""
    lines = ["**Sources**:\n"]
    for i, sws in enumerate(response.source_nodes[:top_n], 1):
        node = getattr(sws, "node", sws)
        score = getattr(sws, "score", None)
        meta = getattr(node, "metadata", {}) or {}
        fp = meta.get("file_path", meta.get("id_", "?"))
        # Short snippet
        snippet = ""
        try:
            snippet = node.get_content(metadata_mode="none")[:240].replace("\n", " ")
        except Exception:
            snippet = (getattr(node, "text", "") or "")[:240].replace("\n", " ")
        if score is None:
            lines.append(f"> {i}. `{fp}` — {snippet} …\n")
        else:
            lines.append(f"> {i}. `{fp}` (score={score:.3f}) — {snippet} …\n")
    return "\n".join(lines)


def query_index(
    persist_dir: Path,
    query_str: str,
    k: int = 5,
    citations: bool = True,
    hybrid: bool = True,
    format_: str = "md",
) -> Dict[str, Any]:
    load_dotenv(".env")

    # Ensure LLM + the SAME embedding backend used at index time (read from manifest)
    _setup_llm()
    embed_spec = _load_manifest_embed(persist_dir)
    setup_embeddings_from_string(embed_spec)

    # Optional reranker from profile
    reranker_name = _load_manifest_reranker(persist_dir)
    reranker = _maybe_make_reranker(reranker_name)

    index = _load_index(persist_dir)

    # Vector + citations engine (attach reranker if available)
    node_postprocessors = []
    if reranker is not None:
        node_postprocessors.append(reranker)

    engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=k,
        citation_chunk_size=512,
        node_postprocessors=node_postprocessors,
    )

    # Optional hybrid prefetch (BM25) — still just for debug visibility
    if hybrid:
        nodes = _load_nodes_texts(persist_dir)
        bm25_idxs = _bm25_candidates(query_str, nodes, top_k=max(k, 8))
        bm25_hits = [nodes[i] for i in bm25_idxs]
    else:
        bm25_hits = []

    response = engine.query(query_str)
    text = str(response)

    if format_ == "md":
        body = f"{text}\n\n"
        if citations:
            body += _format_sources_md(response, top_n=k) or ""
        if bm25_hits:
            body += "\n<details><summary>BM25 candidates (debug)</summary>\n\n"
            for h in bm25_hits[:k]:
                fp = h["metadata"].get("file_path", "?")
                body += f"- {fp}\n"
            body += "\n</details>\n"
        return {"format": "md", "content": body}

    elif format_ == "jsonl":
        lines = [{"role": "assistant", "content": text, "type": "answer"}]
        if citations:
            cites = _format_sources_md(response, top_n=k)
            if cites:
                lines.append({"role": "assistant", "content": cites, "type": "citations"})
        return {"format": "jsonl", "content": "\n".join(json.dumps(x) for x in lines)}

    else:
        return {"format": "raw", "content": text}

