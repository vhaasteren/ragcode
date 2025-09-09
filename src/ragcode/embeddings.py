from __future__ import annotations
from typing import Optional
import requests

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding

# OpenAI embeddings (hosted)
try:
    from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore
except Exception:  # pragma: no cover
    OpenAIEmbedding = None  # type: ignore


# --- Ollama adapter -----------------------------------------------------------

class OllamaEmbedding(BaseEmbedding):
    """
    Minimal embedding adapter for Ollama's /api/embeddings.
    """

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def _embed_batch(self, texts):
        out = []
        for t in texts:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=120,
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out

    def get_query_embedding(self, query: str):
        return self._embed_batch([query])[0]

    def get_text_embedding(self, text: str):
        return self._embed_batch([text])[0]

    def get_text_embeddings(self, texts):
        return self._embed_batch(texts)


# --- Setup helpers ------------------------------------------------------------

def setup_embeddings_from_string(embed_spec: Optional[str]) -> None:
    """
    Configure LlamaIndex Settings.embed_model from a spec string:
      - "openai:text-embedding-3-large"
      - "local:bge-base-en-v1.5"
      - "ollama:nomic-embed-text"
    If None/empty or unknown, leave as-is (LlamaIndex default).
    """
    if not embed_spec:
        return

    if embed_spec.startswith("openai:"):
        model = embed_spec.split("openai:", 1)[1] or "text-embedding-3-large"
        if OpenAIEmbedding is None:
            # If the package isn't installed, we simply leave Settings.embed_model unset.
            return
        Settings.embed_model = OpenAIEmbedding(model=model)
        return

    if embed_spec.startswith("local:"):
        model = embed_spec.split("local:", 1)[1]
        # Lazy import to avoid heavy import when unused
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
        except Exception:
            return
        Settings.embed_model = HuggingFaceEmbedding(model_name=model)
        return

    if embed_spec.startswith("ollama:"):
        model = embed_spec.split("ollama:", 1)[1] or "nomic-embed-text"
        Settings.embed_model = OllamaEmbedding(model=model)
        return

    # Unknown spec: do nothing (fallback to default)
    return
