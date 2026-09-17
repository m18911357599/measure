#!/usr/bin/env python3
"""Download 26 Pattern CANN benchmark kernels from gitcode.com/cann into this repo.

Default dest: current measure checkout (D:\\cursor\\measure on Windows).
Layout matches local_ops discovery: ops-nn / ops-math / ops-transformer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = Path(__file__).resolve().parent / "cann_ops_map.json"
REPOS = ("ops-nn", "ops-math", "ops-transformer")
KEEP_SUBDIRS = ("op_kernel",)
KEEP_FILES = ("CMakeLists.txt", "README.md", "readme.md")
REPO_META = ("LICENSE", "README.md", "README_en.md")


def run(cmd, cwd=None, check=True):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check)


def load_map() -> dict:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def dirs_by_repo(mapping: dict) -> dict:
    out = {r: [] for r in REPOS}
    for _id, rec in mapping["picks"].items():
        for d in rec.get("dirs") or []:
            repo, rest = d.split("/", 1)
            out.setdefault(repo, []).append(rest)
    for r in out:
        out[r] = sorted(set(out[r]))
    return out


def sparse_clone(repo: str, dest: Path, tag: str, paths: list) -> None:
    url = f"https://gitcode.com/cann/{repo}.git"
    if dest.exists() and (dest / ".git").exists():
        run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", tag])
        run(["git", "-C", str(dest), "checkout", "--force", "FETCH_HEAD"])
    else:
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        run([
            "git", "clone", "--filter=blob:none", "--sparse", "--depth", "1",
            "-b", tag, url, str(dest),
        ])
    run(["git", "-C", str(dest), "sparse-checkout", "init", "--cone"])
    if paths:
        run(["git", "-C", str(dest), "sparse-checkout", "set", "--cone", *paths])


def copy_op(src_op: Path, dst_op: Path) -> int:
    """Copy kernel sources; skip tests/examples/docs to keep size down."""
    if not src_op.exists():
        return 0
    dst_op.mkdir(parents=True, exist_ok=True)
    n = 0
    kernel = src_op / "op_kernel"
    if kernel.is_dir():
        dest_k = dst_op / "op_kernel"
        if dest_k.exists():
            shutil.rmtree(dest_k)
        shutil.copytree(kernel, dest_k, ignore=shutil.ignore_patterns("*.o", "*.a", "build"))
        n += sum(1 for p in dest_k.rglob("*") if p.is_file())
    else:
        # no kernel: copy code files only
        for p in src_op.rglob("*"):
            if not p.is_file():
                continue
            if any(x in p.parts for x in ("tests", "examples", "docs", ".git")):
                continue
            if p.suffix.lower() not in {".cpp", ".h", ".hpp", ".c", ".cc", ".cce", ".inl", ".txt", ".md"}:
                continue
            rel = p.relative_to(src_op)
            out = dst_op / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            n += 1
    for fn in KEEP_FILES:
        f = src_op / fn
        if f.is_file():
            shutil.copy2(f, dst_op / f.name)
            n += 1
    return n


def copy_repo_meta(clone_root: Path, dest_root: Path) -> None:
    """Keep CANN Open Software License with the vendored kernels."""
    for repo in REPOS:
        src = clone_root / repo
        dst = dest_root / repo
        dst.mkdir(parents=True, exist_ok=True)
        for fn in REPO_META:
            f = src / fn
            if f.is_file():
                shutil.copy2(f, dst / fn)


def materialize(clone_root: Path, dest_root: Path, mapping: dict) -> dict:
    copy_repo_meta(clone_root, dest_root)
    stats = {}
    for _id, rec in mapping["picks"].items():
        files = 0
        present = []
        for d in rec.get("dirs") or []:
            src = clone_root / d
            dst = dest_root / d
            n = copy_op(src, dst)
            files += n
            if n:
                present.append(d)
        stats[_id] = {
            "pattern": rec["pattern"],
            "files": files,
            "present": present,
            "missing_note": rec.get("missing", ""),
        }
    return stats


def write_stamp(dest_root: Path, mapping: dict, stats: dict) -> None:
    stamp = {
        "tag": mapping["tag"],
        "source": mapping["source"],
        "license": "CANN Open Software License Agreement Version 2.0",
        "operators": stats,
    }
    (dest_root / "cann_ops_downloaded.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(ROOT), help="measure repo root (default: this checkout)")
    ap.add_argument("--work", default="", help="clone cache dir (default: dest/.cann-src)")
    ap.add_argument("--tag", default="", help="override git tag/branch")
    args = ap.parse_args(argv)
    dest = Path(args.dest).expanduser().resolve()
    mapping = load_map()
    tag = args.tag or mapping["tag"]
    work = Path(args.work).expanduser().resolve() if args.work else dest / ".cann-src"
    by_repo = dirs_by_repo(mapping)
    print("dest", dest)
    print("tag ", tag)
    for repo, paths in by_repo.items():
        print(f"clone {repo} ({len(paths)} dirs)")
        sparse_clone(repo, work / repo, tag, paths)
    stats = materialize(work, dest, mapping)
    write_stamp(dest, mapping, stats)
    ok = sum(1 for s in stats.values() if s["files"] > 0)
    print(f"downloaded {ok}/26 patterns with source")
    for i in sorted(stats, key=lambda x: int(x)):
        s = stats[i]
        flag = "OK" if s["files"] else "EMPTY"
        print(f"  [{flag}] #{i:>2} {s['pattern']}: {s['files']} files {s['present']}")
        if s["missing_note"] and not s["files"]:
            print("         ", s["missing_note"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
