from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
from .logging import info, warn

def _load_symbols(persist_dir: Path) -> List[Dict[str, Any]]:
    p = persist_dir / "symbols.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def explain_symbol(persist_dir: Path, symbol: str) -> str:
    """Best-effort: show where it's defined and provide context lines from the chunk store."""
    symbols = _load_symbols(persist_dir)
    hits = [s for s in symbols if s["symbol"] == symbol]
    if not hits:
        return f"No symbol named '{symbol}' in symbols index."
    # Render simple report
    out = [f"# Explanation for `{symbol}` (definitions)"]
    for h in hits[:5]:
        out.append(f"- {h['kind']} at `{h['file_path']}:{h['start_line']}-{h['end_line']}`")
    out.append("\nTip: run `ragcode query \"Explain {symbol} in context\" --k 8 --citations`")
    return "\n".join(out)

def where_symbol(persist_dir: Path, symbol: str) -> str:
    symbols = _load_symbols(persist_dir)
    hits = [s for s in symbols if s["symbol"] == symbol]
    if not hits:
        return f"No symbol named '{symbol}'."
    out = [f"# Locations for `{symbol}`"]
    for h in hits:
        out.append(f"- {h['file_path']}:{h['start_line']}-{h['end_line']} ({h['kind']})")
    return "\n".join(out)

