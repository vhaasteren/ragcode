from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional

def extract_python_symbols(path: Path, text: str) -> List[Dict[str, Any]]:
    """Very lightweight symbol extractor for Python (functions/classes)."""
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", lineno)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            out.append({
                "symbol": name,
                "kind": kind,
                "start_line": lineno,
                "end_line": end_lineno,
                "file_path": str(path),
                "language": "python",
            })
    return out

