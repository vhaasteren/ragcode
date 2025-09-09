from __future__ import annotations
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, field_validator

from .config import load_profile
from .query import query_index
from .explain_where import explain_symbol, where_symbol


app = FastAPI(title="ragcode API", version="0.1.1")


class QueryReq(BaseModel):
    # You may provide persist_dir directly...
    persist_dir: Optional[str] = None
    # ...or provide a profile to resolve persist_dir
    profile: Optional[str] = None
    local_config: Optional[str] = None

    query: str
    k: int = 5
    citations: bool = True
    hybrid: bool = True
    format: str = "md"

    @field_validator("format")
    @classmethod
    def _fmt(cls, v: str) -> str:
        v = (v or "md").lower()
        if v not in {"md", "jsonl", "raw"}:
            raise ValueError("format must be one of: md, jsonl, raw")
        return v


class ExplainReq(BaseModel):
    persist_dir: Optional[str] = None
    profile: Optional[str] = None
    local_config: Optional[str] = None
    symbol: str


def _resolve_persist_dir(persist_dir: Optional[str], profile: Optional[str], local_config: Optional[str]) -> Path:
    if persist_dir:
        return Path(persist_dir).expanduser()
    if profile:
        lc_path = Path(local_config).expanduser() if local_config else Path(".ragcode.toml")
        prof = load_profile(profile_name=profile, local_config=lc_path)
        return Path(prof.persist).expanduser()
    raise HTTPException(status_code=422, detail="Either 'persist_dir' or 'profile' must be provided.")


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "name": "ragcode API",
        "version": "0.1.1",
        "endpoints": {
            "query": "POST /query",
            "explain": "POST /explain",
            "where": "POST /where",
            "inspect": "POST /inspect",
            "docs": "GET /docs"
        },
        "hint": "Open /docs for the interactive Swagger UI."
    })


@app.get("/healthz", include_in_schema=False)
def healthz():
    return PlainTextResponse("ok")


@app.post("/query")
def api_query(req: QueryReq):
    persist = _resolve_persist_dir(req.persist_dir, req.profile, req.local_config)
    return query_index(persist, req.query, req.k, req.citations, req.hybrid, req.format)


@app.post("/explain")
def api_explain(req: ExplainReq):
    persist = _resolve_persist_dir(req.persist_dir, req.profile, req.local_config)
    return {"format": "md", "content": explain_symbol(persist, req.symbol)}


@app.post("/where")
def api_where(req: ExplainReq):
    persist = _resolve_persist_dir(req.persist_dir, req.profile, req.local_config)
    return {"format": "md", "content": where_symbol(persist, req.symbol)}


# Optional helper to print manifest via API
class InspectReq(BaseModel):
    persist_dir: Optional[str] = None
    profile: Optional[str] = None
    local_config: Optional[str] = None

@app.post("/inspect")
def api_inspect(req: InspectReq):
    persist = _resolve_persist_dir(req.persist_dir, req.profile, req.local_config)
    manifest = persist / "manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail=f"manifest.json not found in {persist}")
    return JSONResponse((manifest.read_text(encoding="utf-8")))

