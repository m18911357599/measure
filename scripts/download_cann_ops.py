#!/usr/bin/env python3
"""Download 26 Pattern benchmark kernels into this repo.

AscendC: gitcode.com/cann ops-nn / ops-math / ops-transformer
SuperScalar: github.com/PTO-ISA/SuperNpuBench @ ops-20260908
  benchmark/one-level-arch/kernels/solution (+ single_thread fills)
SIMT: local D:/simt (preferred) or gitcode.com/mlidongfeng/gemm-cuda

Default dest: current measure checkout (D:\\cursor\\measure on Windows).
Private gitcode repos: set GITCODE_TOKEN to a gitcode personal access token.
Local SIMT: python scripts/download_cann_ops.py --skip-cann --skip-super --simt-local D:/simt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = Path(__file__).resolve().parent / "cann_ops_map.json"
SUPER_MAP_PATH = Path(__file__).resolve().parent / "super_ops_map.json"
REPOS = ("ops-nn", "ops-math", "ops-transformer")
KEEP_SUBDIRS = ("op_kernel",)
KEEP_FILES = ("CMakeLists.txt", "README.md", "readme.md")
REPO_META = ("LICENSE", "README.md", "README_en.md")
SUPER_URL = "https://github.com/PTO-ISA/SuperNpuBench.git"
SUPER_DEST_NAME = "SuperNpuBench"
SIMT_MAP_PATH = Path(__file__).resolve().parent / "simt_ops_map.json"
SIMT_URL = "https://gitcode.com/mlidongfeng/gemm-cuda.git"
SIMT_DEST_NAME = "simt"
SIMT_LOCAL_DEFAULT = "D:/simt"
SIMT_SOURCE_EXTS = {".cu", ".cuh", ".h", ".hpp", ".hh", ".cpp", ".cc", ".c", ".inl"}
SIMT_KEEP_EXTS = SIMT_SOURCE_EXTS | {".txt", ".md", ".rst", ".cmake", ".py"}
SIMT_KEEP_NAMES = {
    "CMakeLists.txt", "Makefile", "LICENSE", "LICENSE.txt", "README.md", "readme.md",
}
SIMT_SKIP_DIRS = {".git", "build", "__pycache__", ".github", ".ci", "node_modules"}
SIMT_OPTIONAL_SKIP = {"tests", "test", "docs", "examples", "third_party", "3rdparty"}
GEMM_FALLBACK_ID = "4"
GEMM_NAME_HINTS = ("gemm", "matmul", "sgemm", "hgemm", "wmma", "mma")


def run(cmd, cwd=None, check=True, env=None):
    print("+", " ".join(_redact_cmd(cmd)))
    return subprocess.run(cmd, cwd=cwd, check=check, env=env)


def _redact_cmd(cmd):
    out = []
    for c in cmd:
        if isinstance(c, str) and "://" in c and "@" in c:
            out.append(re.sub(r"://([^/@]+):([^/@]+)@", r"://\1:***@", c))
        else:
            out.append(str(c))
    return out


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


def load_super_map() -> dict:
    return json.loads(SUPER_MAP_PATH.read_text(encoding="utf-8"))


def sparse_clone_url(url: str, dest: Path, tag: str, paths: list) -> None:
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


def copy_tree_files(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("*.o", "*.a", "build", ".git", "__pycache__"),
    )
    return sum(1 for p in dst.rglob("*") if p.is_file())


def materialize_super(clone_root: Path, dest_root: Path, mapping: dict) -> dict:
    rel_root = mapping["root"]
    src_kernels = clone_root / rel_root
    dst_kernels = dest_root / SUPER_DEST_NAME / rel_root
    readme = clone_root / "README.md"
    dst_repo = dest_root / SUPER_DEST_NAME
    dst_repo.mkdir(parents=True, exist_ok=True)
    if readme.is_file():
        shutil.copy2(readme, dst_repo / "README.md")
    kr = src_kernels / "README.md"
    if kr.is_file():
        dst_kernels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kr, dst_kernels / "README.md")
    stats = {}
    for _id, rec in mapping["picks"].items():
        files = 0
        present = []
        for d in rec.get("dirs") or []:
            src = src_kernels / d
            dst = dst_kernels / d
            n = copy_tree_files(src, dst)
            files += n
            if n:
                present.append(f"{SUPER_DEST_NAME}/{rel_root}/{d}")
        stats[_id] = {
            "pattern": rec["pattern"],
            "files": files,
            "present": present,
            "missing_note": rec.get("missing", ""),
        }
    # Keep solution-level README / leftover solution kernels already in picks.
    st_readme = src_kernels / "single_thread" / "README.md"
    if st_readme.is_file() and (dst_kernels / "single_thread").is_dir():
        shutil.copy2(st_readme, dst_kernels / "single_thread" / "README.md")
    return stats


def write_super_stamp(dest_root: Path, mapping: dict, stats: dict) -> None:
    stamp = {
        "branch": mapping["branch"],
        "source": mapping["source"],
        "root": f"{SUPER_DEST_NAME}/{mapping['root']}",
        "operators": stats,
    }
    (dest_root / "super_ops_downloaded.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_simt_map() -> dict:
    if SIMT_MAP_PATH.is_file():
        return json.loads(SIMT_MAP_PATH.read_text(encoding="utf-8"))
    return {
        "source": SIMT_URL.rstrip(".git"),
        "upstream": "https://gitcode.com/LinPX/gemm-cuda",
        "dest": SIMT_DEST_NAME,
        "branch": "",
    }


def simt_auth_url(url: str, token: str, username: str = "oauth2") -> str:
    """Embed a gitcode PAT in an HTTPS git URL. Never log the result."""
    token = (token or "").strip()
    if not token:
        return url
    user = (username or "oauth2").strip() or "oauth2"
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" in rest.split("/", 1)[0]:
        rest = rest.split("@", 1)[1]
    return f"{scheme}://{user}:{token}@{rest}"


class SimtCloneError(RuntimeError):
    """Clone of the SIMT gemm-cuda repo failed."""


def clone_simt(
    url: str,
    dest: Path,
    token: str = "",
    username: str = "oauth2",
    branch: str = "",
) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    auth = simt_auth_url(url, token, username) if token else url
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["-b", branch]
    cmd += [auth, str(dest)]
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    print("+", " ".join(_redact_cmd(cmd)))
    r = subprocess.run(
        cmd, env=env, check=False, text=True, capture_output=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        err = re.sub(r"://([^/@]+):([^/@]+)@", r"://\1:***@", err)
        raise SimtCloneError(err or f"git clone failed with code {r.returncode}")


def _simt_rel_skipped(rel: Path, extra_skip: set) -> bool:
    parts = rel.parts[:-1]
    skip = SIMT_SKIP_DIRS | extra_skip
    return any(part in skip or part.startswith(".") for part in parts)


def copy_simt_tree(
    src: Path,
    dst: Path,
    include_optional: bool = False,
    extra_skip: set | None = None,
) -> int:
    """Copy CUDA/C++ kernel sources and docs; skip .git/build.

    extra_skip overrides the default optional skips (tests/docs/third_party).
    """
    if not src.exists():
        return 0
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if extra_skip is not None:
        extra = set(extra_skip)
    else:
        extra = set() if include_optional else set(SIMT_OPTIONAL_SKIP)
    n = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if _simt_rel_skipped(rel, extra):
            continue
        keep = p.name in SIMT_KEEP_NAMES or p.suffix.lower() in SIMT_KEEP_EXTS
        if not keep:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def _count_simt_source(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SIMT_SOURCE_EXTS
    )


def vendor_simt_from_local(src: Path, dest_root: Path) -> Path:
    """Copy D:/simt (or any local folder) into dest/simt for git pull + extract."""
    dst = dest_root / SIMT_DEST_NAME
    n = copy_simt_tree(src, dst, extra_skip={"third_party", "3rdparty"})
    if _count_simt_source(dst) == 0:
        n = copy_simt_tree(src, dst, extra_skip=set())
    print(f"imported {n} SIMT files from {src} -> {dst}")
    return dst


def find_simt_local(explicit: str = "") -> Optional[Path]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from local_ops import expand_local_path  # noqa: WPS433

    cands = []
    if explicit:
        cands.append(explicit)
    else:
        cands.extend([
            os.environ.get("SIMT_ROOT") or "",
            os.environ.get("MEASURE_SIMT") or "",
            SIMT_LOCAL_DEFAULT,
            "d:/simt",
        ])
    for raw in cands:
        p = expand_local_path(raw)
        if p is not None and p.is_dir():
            return p
    return None


def materialize_simt(clone_root: Path, dest_root: Path) -> Path:
    dst = dest_root / SIMT_DEST_NAME
    n = copy_simt_tree(clone_root, dst, include_optional=False)
    if _count_simt_source(dst) == 0:
        n = copy_simt_tree(clone_root, dst, include_optional=True)
    print(f"copied {n} SIMT files -> {dst}")
    return dst


def map_simt_patterns(simt_root: Path, operators: dict) -> dict:
    """Classify vendored SIMT files into the 26 patterns."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_source import match_by_tokens, match_globs  # noqa: WPS433

    assigned = set()
    stats = {}
    ops = operators.get("operators") or []
    exts = ((operators.get("source_roots") or {}).get("SIMT") or {}).get("extensions") or list(SIMT_SOURCE_EXTS)
    for op in ops:
        oid = str(op["id"])
        globs = (op.get("globs") or {}).get("SIMT") or []
        files = match_globs(simt_root, globs, "SIMT")
        if not files:
            tokens = []
            for part in re.split(r"[,/·\s]+", str(op.get("op") or "")):
                if part:
                    tokens.append(part)
            pat = str(op.get("pattern") or "")
            specific = pat.split("·", 1)[1] if "·" in pat else ""
            for part in re.split(r"[,/·\s]+", specific):
                if part and len(part) >= 3:
                    tokens.append(part)
            files = match_by_tokens(simt_root, tokens, exts, "SIMT")
        files = [f for f in files if f.suffix.lower() in {e.lower() for e in exts}]
        rels = []
        for f in files:
            try:
                rel = str(f.relative_to(simt_root)).replace("\\", "/")
            except ValueError:
                rel = str(f)
            assigned.add(rel)
            rels.append(rel)
        stats[oid] = {
            "pattern": op.get("pattern"),
            "files": len(rels),
            "present": rels[:20],
            "missing_note": "" if rels else "no matching SIMT kernel in simt/",
        }

    leftovers = []
    for p in simt_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SIMT_SOURCE_EXTS:
            continue
        try:
            rel = str(p.relative_to(simt_root)).replace("\\", "/")
        except ValueError:
            continue
        if rel in assigned:
            continue
        name = rel.lower().replace("\\", "/")
        if any(h in name.replace("_", "").replace("-", "") for h in GEMM_NAME_HINTS):
            leftovers.append(rel)
    if leftovers:
        rec = stats.get(GEMM_FALLBACK_ID)
        if rec is not None:
            rec["present"] = list(dict.fromkeys(list(rec.get("present") or []) + leftovers))
            rec["files"] = len(rec["present"])
            rec["missing_note"] = ""
    return stats


def write_simt_stamp(dest_root: Path, mapping: dict, stats: dict, extra: dict) -> None:
    stamp = {
        "source": mapping.get("source"),
        "upstream": mapping.get("upstream"),
        "dest": SIMT_DEST_NAME,
        "operators": stats,
    }
    stamp.update(extra)
    (dest_root / "simt_ops_downloaded.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_stats(title: str, stats: dict) -> int:
    ok = sum(1 for s in stats.values() if s["files"] > 0)
    print(f"{title}: {ok}/26 patterns with source")
    for i in sorted(stats, key=lambda x: int(x)):
        s = stats[i]
        flag = "OK" if s["files"] else "EMPTY"
        print(f"  [{flag}] #{i:>2} {s['pattern']}: {s['files']} files {s['present']}")
        if s["missing_note"] and not s["files"]:
            print("         ", s["missing_note"])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(ROOT), help="measure repo root (default: this checkout)")
    ap.add_argument("--work", default="", help="clone cache dir (default: dest/.cann-src)")
    ap.add_argument("--tag", default="", help="override CANN git tag/branch")
    ap.add_argument("--skip-cann", action="store_true", help="Do not clone gitcode.com/cann")
    ap.add_argument("--skip-super", action="store_true", help="Do not clone SuperNpuBench")
    ap.add_argument("--skip-simt", action="store_true", help="Do not import or clone SIMT kernels")
    ap.add_argument(
        "--simt-local", default="",
        help="Import SIMT operators from a local folder (e.g. D:/simt). "
             "If omitted, D:/simt is used when it exists, else git clone.",
    )
    ap.add_argument(
        "--simt-from-git", action="store_true",
        help="Clone gitcode.com/mlidongfeng/gemm-cuda even if D:/simt exists",
    )
    ap.add_argument("--simt-url", default="", help="Override SIMT git URL")
    ap.add_argument("--simt-branch", default="", help="Override SIMT git branch/tag")
    ap.add_argument(
        "--gitcode-token", default="",
        help="gitcode PAT (else env GITCODE_TOKEN). Not printed.",
    )
    args = ap.parse_args(argv)
    dest = Path(args.dest).expanduser().resolve()
    work = Path(args.work).expanduser().resolve() if args.work else dest / ".cann-src"
    rc = 0

    print("dest", dest)
    if not args.skip_cann:
        mapping = load_map()
        tag = args.tag or mapping["tag"]
        by_repo = dirs_by_repo(mapping)
        print("cann tag", tag)
        for repo, paths in by_repo.items():
            print(f"clone {repo} ({len(paths)} dirs)")
            sparse_clone(repo, work / repo, tag, paths)
        stats = materialize(work, dest, mapping)
        write_stamp(dest, mapping, stats)
        if print_stats("CANN/AscendC", stats) == 0:
            rc = 1

    if not args.skip_super:
        smap = load_super_map()
        branch = smap["branch"]
        super_work = dest / ".super-src"
        clone = super_work / SUPER_DEST_NAME
        print("super branch", branch)
        sparse_clone_url(SUPER_URL, clone, branch, [smap["root"]])
        sstats = materialize_super(clone, dest, smap)
        write_super_stamp(dest, smap, sstats)
        if print_stats("SuperScalar", sstats) == 0:
            rc = 1

    if not args.skip_simt:
        smap = load_simt_map()
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from extract_source import load_operators  # noqa: WPS433

        def _write_simt(simt_dst: Path, imported_from: str) -> int:
            operators = load_operators()
            tstats = map_simt_patterns(simt_dst, operators)
            write_simt_stamp(dest, smap, tstats, {
                "status": "ok",
                "root": SIMT_DEST_NAME,
                "imported_from": imported_from,
            })
            return 1 if print_stats("SIMT", tstats) == 0 else 0

        local_src = None
        if not args.simt_from_git:
            local_src = find_simt_local(args.simt_local)
            if args.simt_local and local_src is None:
                print(f"SIMT local path not found: {args.simt_local}")
                print("Place kernels at D:/simt or pass --simt-local <dir>.")
                rc = 1
            elif local_src is not None:
                print("simt local", local_src)
                simt_dst = vendor_simt_from_local(local_src, dest)
                if _write_simt(simt_dst, str(local_src).replace("\\", "/")):
                    rc = 1

        if local_src is None and not (args.simt_local and not args.simt_from_git):
            url = args.simt_url or SIMT_URL
            branch = args.simt_branch or smap.get("branch") or ""
            token = args.gitcode_token or os.environ.get("GITCODE_TOKEN") or os.environ.get("GITCODE_PRIVATE_TOKEN") or ""
            username = os.environ.get("GITCODE_USERNAME") or "oauth2"
            simt_work = dest / ".simt-src" / SIMT_DEST_NAME
            print("simt", url, "branch", branch or "(default)")
            try:
                clone_simt(url, simt_work, token=token, username=username, branch=branch)
            except SimtCloneError as exc:
                msg = str(exc)
                print("SIMT clone failed:", msg)
                if "Authentication failed" in msg or "Access denied" in msg or "could not read Username" in msg or "401" in msg:
                    print(
                        "gitcode.com/mlidongfeng/gemm-cuda is private. "
                        "Import local kernels instead:\n"
                        "  python scripts/download_cann_ops.py --skip-cann --skip-super --simt-local D:/simt\n"
                        "Or set GITCODE_TOKEN and retry."
                    )
                if args.skip_cann and args.skip_super:
                    rc = 1
                else:
                    print("continuing without SIMT kernels")
            else:
                simt_dst = materialize_simt(simt_work, dest)
                if _write_simt(simt_dst, url):
                    rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
