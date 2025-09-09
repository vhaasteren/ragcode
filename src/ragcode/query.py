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

def _setup_llm():
    Settings.llm = OpenAI(model="gpt-4o-mini")

def _load_index(persist_dir: Path):
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    return index

def _load_manifest_embed(persist_dir: Path) -> Optional[str]:
    mf = persist_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
        prof = data.get("profile") or {}
        return prof.get("embed")
    except Exception:
        return None

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
        text = node.get("text") or node.get("node", {}).get("text") or ""
        meta = node.get("metadata", {}) or node.get("node", {}).get("metadata", {})
        result.append({"text": text, "metadata": meta, "ref_id": node.get("id_", "")})
    return result

def _bm25_candidates(query: str, nodes: List[Dict[str, Any]], top_k: int) -> List[int]:
    if not nodes:
        return []
    tokenized_corpus = [n["text"].split() for n in nodes]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    # Return top_k indices
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return idxs

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

    index = _load_index(persist_dir)

    # Vector + citations engine
    engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=k,
        citation_chunk_size=512,
        node_postprocessors=[],
    )

    # Optional hybrid prefetch (BM25)
    if hybrid:
        nodes = _load_nodes_texts(persist_dir)
        bm25_idxs = _bm25_candidates(query_str, nodes, top_k=max(k, 8))
        # Not directly pluggable; we just log what BM25 thought was good.
        bm25_hits = [nodes[i] for i in bm25_idxs]
    else:
        bm25_hits = []

    response = engine.query(query_str)
    text = str(response)
    sources = response.get_formatted_sources() if citations else ""

    if format_ == "md":
        body = f"{text}\n\n"
        if citations and sources:
            body += f"**Sources**:\n\n{sources}\n"
        if bm25_hits:
            body += "\n<details><summary>BM25 candidates (debug)</summary>\n\n"
            for h in bm25_hits[:k]:
                fp = h["metadata"].get("file_path", "?")
                body += f"- {fp}\n"
            body += "\n</details>\n"
        return {"format": "md", "content": body}

    elif format_ == "jsonl":
        lines = []
        lines.append({"role": "assistant", "content": text, "type": "answer"})
        if citations and sources:
            lines.append({"role": "assistant", "content": sources, "type": "citations"})
        return {"format": "jsonl", "content": "\n".join(json.dumps(x) for x in lines)}

    else:
        return {"format": "raw", "content": text}

