#!/usr/bin/env python3
"""Discover local cache roots for Windows/Linux measure checkouts.

Default cache: D:/cursor/measure (Cursor local workspace). Operator trees
are optional siblings or subfolders (ops-nn, SuperNpuBench, cuda). If they
are missing, tests/fixtures is used so 度量 still runs on this repo alone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CFG_NAME = "measure_local.json"
DEFAULT_CACHE = "D:/cursor/measure"

ASCEND_HINTS = ("ops-nn", "ops-math", "ops-transformer")
SUPER_HINTS = (
    "benchmark/one-level-arch/kernels/solution",
    "one-level-arch/kernels/solution",
    "kernels/solution",
)
SIMT_HINTS = ("cuda", "cutlass", "cub", "cuda-kernels")


def expand_local_path(raw: Optional[str]) -> Optional[Path]:
    """Parse a user path, including Windows D:\\... forms on any OS."""
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        return None
    s = s.replace("\\", "/")
    drive = len(s) >= 2 and s[1] == ":" and s[0].isalpha()
    if drive:
        p = Path(s)
        if os.name != "nt" and not p.exists():
            return None
        return p.expanduser()
    p = Path(s).expanduser()
    return p


def _exists_file(root: Path, name: str) -> bool:
    return (root / name).is_file()


def load_local_cfg(repo_root: Optional[Path] = None) -> dict:
    root = repo_root or REPO_ROOT
    for candidate in (root / LOCAL_CFG_NAME, Path.cwd() / LOCAL_CFG_NAME):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def cache_candidates(repo_root: Optional[Path] = None) -> List[Path]:
    root = repo_root or REPO_ROOT
    cfg = load_local_cfg(root)
    out: List[Path] = []
    seen = set()

    def add(raw):
        p = expand_local_path(str(raw) if raw is not None else "")
        if p is None:
            return
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    add(os.environ.get("MEASURE_CACHE"))
    add(cfg.get("cache_root"))
    add(DEFAULT_CACHE)
    add("d:/cursor/measure")
    add(root)
    add(Path.cwd())
    return out


def find_cache_root(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or REPO_ROOT
    for p in cache_candidates(root):
        if not p.exists() or not p.is_dir():
            continue
        if _exists_file(p, "measure.md") or _exists_file(p, "report.html") or (p / "scripts" / "measure.py").is_file():
            return p.resolve() if p.exists() else p
    return root.resolve()


def _find_named_dir(base: Path, names: Sequence[str], max_depth: int = 3) -> Optional[Path]:
    names_l = {n.lower() for n in names}
    if not base.exists():
        return None
    if base.name.lower() in names_l:
        return base
    try:
        for child in base.iterdir():
            if child.is_dir() and child.name.lower() in names_l:
                return child
    except OSError:
        return None
    if max_depth <= 0:
        return None
    try:
        children = [c for c in base.iterdir() if c.is_dir() and c.name not in {".git", "node_modules", "__pycache__"}]
    except OSError:
        return None
    for child in children:
        hit = _find_named_dir(child, names, max_depth - 1)
        if hit:
            return hit
    return None


def _find_rel_dir(base: Path, rels: Sequence[str]) -> Optional[Path]:
    for rel in rels:
        p = base.joinpath(*rel.split("/"))
        if p.is_dir():
            return p
    # SuperNpuBench nested one extra level
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            for rel in rels:
                p = child.joinpath(*rel.split("/"))
                if p.is_dir():
                    return p
    except OSError:
        pass
    return None


def discover_roots(cache_root: Optional[Path] = None, repo_root: Optional[Path] = None) -> Dict[str, str]:
    """Return {arch: path} for local extract. Missing trees fall back to fixtures."""
    repo = repo_root or REPO_ROOT
    cfg = load_local_cfg(repo)
    cache = cache_root or find_cache_root(repo)
    fixtures = None
    for cand in (cache / "tests" / "fixtures", repo / "tests" / "fixtures"):
        if cand.is_dir():
            fixtures = cand
            break

    roots: Dict[str, str] = {}
    cfg_roots = cfg.get("roots") or {}

    def configured(arch: str) -> Optional[Path]:
        p = expand_local_path(cfg_roots.get(arch))
        if p is not None and p.exists():
            return p
        return None

    ascend = configured("AscendC")
    if ascend is None:
        found = []
        for name in ASCEND_HINTS:
            hit = _find_named_dir(cache, (name,), 3)
            if hit:
                found.append(hit)
        if len(found) >= 2:
            ascend = found[0].parent
        elif found:
            ascend = found[0]
        elif fixtures:
            ascend = fixtures
        else:
            ascend = cache
    roots["AscendC"] = str(ascend)

    super_p = configured("SuperScalar")
    if super_p is None:
        super_p = _find_rel_dir(cache, SUPER_HINTS)
        if super_p is None:
            super_p = _find_named_dir(cache, ("SuperNpuBench", "supernpubench"), 2)
        if super_p is None:
            super_p = fixtures or cache
    roots["SuperScalar"] = str(super_p)

    simt = configured("SIMT")
    if simt is None:
        simt = _find_named_dir(cache, SIMT_HINTS, 2)
        if simt is None:
            simt = fixtures or cache
    roots["SIMT"] = str(simt)

    return roots


def local_status(repo_root: Optional[Path] = None) -> dict:
    repo = repo_root or REPO_ROOT
    cache = find_cache_root(repo)
    roots = discover_roots(cache, repo)
    existing = {}
    for arch, path in roots.items():
        p = expand_local_path(path) or Path(path)
        existing[arch] = p.exists() if p else False
    return {
        "cache_root": str(cache),
        "repo_root": str(repo.resolve()),
        "default_cache": DEFAULT_CACHE,
        "cfg": LOCAL_CFG_NAME,
        "roots": roots,
        "exists": existing,
        "docs": {
            "measure_md": str((cache / "measure.md")) if (cache / "measure.md").is_file() else str(repo / "measure.md"),
            "report": str((cache / "report.html")) if (cache / "report.html").is_file() else str(repo / "report.html"),
        },
    }
