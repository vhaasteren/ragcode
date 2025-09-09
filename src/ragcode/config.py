from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import tomlkit
from dotenv import load_dotenv

DEFAULT_PERSIST = Path.home() / ".ragcode" / "indexes"

class Profile(BaseModel):
    name: str = "default"
    repo: Optional[str] = None        # e.g., "nanograv/PINT"
    ref: str = "master"
    path: Optional[str] = None        # local clone path
    include: List[str] = Field(default_factory=lambda: ["src", "docs", "examples", "tests"])
    ext: List[str] = Field(default_factory=lambda: [".py", ".md", ".rst", ".txt"])
    persist: str = str(DEFAULT_PERSIST / "default")
    embed: str = "openai:text-embedding-3-large"
    store: str = "llamaindex"         # future: sqlite/chroma/lance, etc.
    hybrid: bool = True
    reranker: str = "none"
    chunk_lines: int = 80
    chunk_overlap: int = 20
    symbol_chunks: bool = True
    parallel: int = 8
    max_file_size_kb: int = 1024

class Settings(BaseModel):
    profile: Profile
    env: Dict[str, Any] = Field(default_factory=dict)

def _read_toml(path: Path) -> Dict[str, Any]:
    data = tomlkit.parse(path.read_text(encoding="utf-8"))
    return dict(data)

def load_profile(profile_name: Optional[str], local_config: Optional[Path]) -> Profile:
    # Load .env early
    load_dotenv(dotenv_path=Path(".env"), override=False)

    # 1) start with packaged default if profile_name matches a bundled file
    pkg_profile_path = Path(__file__).with_name("profiles") / f"{profile_name or 'default'}.toml"
    base: Dict[str, Any] = _read_toml(pkg_profile_path) if pkg_profile_path.exists() else {}

    # 2) merge user profile from ~/.ragcode/profiles/{name}.toml
    user_profile_dir = Path.home() / ".ragcode" / "profiles"
    user_profile_path = user_profile_dir / f"{profile_name or 'default'}.toml"
    if user_profile_path.exists():
        base.update(_read_toml(user_profile_path))

    # 3) merge project-local .ragcode.toml
    if local_config and local_config.exists():
        base.update(_read_toml(local_config))

    # Fill defaults via pydantic
    return Profile(**base) if base else Profile(name=profile_name or "default")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

