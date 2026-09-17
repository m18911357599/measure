#!/usr/bin/env python3
"""Benchmark operator source extraction for NPU usability metrics.

Scans AscendC (ops-nn / ops-math / ops-transformer), SuperScalar
(SuperNpuBench kernels/solution) and SIMT (CUDA) kernels, then emits
per-operator line/API/complexity counters consumed by score.py / report.html.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_OPERATORS = ROOT / "operators.json"

SKIP_DIR_NAMES = {
    "build", ".git", ".svn", "third_party", "3rdparty", "node_modules",
    "__pycache__", "docs", "examples", "test", "tests", "ut", "st",
    "cmake", "figures",
    # Keep ISA trees from leaking into each other's extract when the root is the repo.
    "SuperNpuBench", "supernpubench", "cuda", "cutlass", "cub",
}

SOURCE_EXTS = {
    ".cpp", ".cc", ".c", ".h", ".hpp", ".cce", ".inl",
    ".cu", ".cuh", ".hip",
}

# Architecture-specific keyword patterns (line-level static analysis).
ARCH_PATTERNS: Dict[str, Dict[str, re.Pattern]] = {
    "AscendC": {
        "dma": re.compile(
            r"DataCopy(?:Pad|2D)?|Copy\s*\(|Nd2Nz|Nz2Nd|SetNdDma|"
            r"TQue|EnQue|DeQue|MTE2|MTE3|GlobalTensor|GM_ADDR",
            re.I,
        ),
        "dma_call": re.compile(r"DataCopy(?:Pad|2D)?\s*\(|Copy\s*\("),
        "coh": re.compile(
            r"DCCI|dcci|Invalidate|Prefetch|cacheline|CacheMode|"
            r"SetCacheMode|dcflush|icache|dcache",
            re.I,
        ),
        "sync_inter": re.compile(
            r"CrossCore(?:Set|Wait)Flag|SyncAll|Ipc(?:Set|Wait)|Notify|"
            r"GetBlockNum|GetSubBlockIdx",
            re.I,
        ),
        "sync_intra": re.compile(
            r"SetFlag|WaitFlag|PipeBarrier|pipe_barrier|event_t|TEventID",
            re.I,
        ),
        "pc": re.compile(
            r"TQue|Producer|Consumer|EnQue|DeQue|GetTPipePtr|TPipe",
            re.I,
        ),
        "scalar": re.compile(
            r"GetBlockIdx|GetBlockNum|block_idx|GetUserWorkspace|"
            r"GetSysWorkSpace|printf|assert|for\s*\(",
            re.I,
        ),
        "alloc": re.compile(
            r"AllocTensor|FreeTensor|InitBuffer|TBuf|TPipe\s+\w+|GetTQueHead",
            re.I,
        ),
        "addr": re.compile(
            r"Gather|Scatter|stride|offset|indexOffset|gIndex|blockOffset|"
            r"dataIndex|startIndex",
            re.I,
        ),
        "hint": re.compile(r"\bhint\b|Prefetch|SetCacheMode|persistent", re.I),
        "cluster": re.compile(r"cluster|GetCluster|dieId|DieId|GetDie", re.I),
        "compute": re.compile(
            r"\b(?:Add|Mul|Muls|Exp|Ln|Abs|Relu|Div|Sub|Max|Min|Mmad|"
            r"Axpy|Cast|SoftMax|ReduceSum)\s*\(|CubeCompute",
        ),
        "pipe_switch": re.compile(
            r"SetFlag|WaitFlag|PipeBarrier|EnQue|DeQue",
            re.I,
        ),
        "scale": re.compile(
            r"scale|per.?channel|per.?block|dequant|antiquant|mxscale",
            re.I,
        ),
        "api_decl": re.compile(
            r"(?:__aicore__|__global__)?\s*(?:inline\s+)?(?:void|auto|"
            r"template\s*<[^>]+>\s*\w+)\s+(\w+)\s*\(",
        ),
    },
    "SuperScalar": {
        "dma": re.compile(
            r"TLOAD(?:_CUBE)?|TSTORE(?:_CUBE)?|TCOPY(?:IN|OUT)?|TMA\b",
        ),
        "dma_call": re.compile(
            r"TLOAD(?:_CUBE)?\s*\(|TSTORE(?:_CUBE)?\s*\(|TCOPY",
        ),
        "coh": re.compile(r"fence|invalidate|cache|dcci", re.I),
        "sync_inter": re.compile(
            r"get_thread_idx|BARRIER|sync_pe|cluster|multi_pe",
            re.I,
        ),
        "sync_intra": re.compile(
            r"\bwait\b|barrier|BENCHSTART|BENCHEND|PipeWait",
            re.I,
        ),
        "pc": re.compile(r"producer|consumer|pipeline|stage", re.I),
        "scalar": re.compile(
            r"\bg[MNK]\b|lda|ldb|ldc|for\s*\(|if\s*\(",
        ),
        "alloc": re.compile(
            r"Tile\s*<|CubeTile|CubeAccumulator|scratch|buffer|global_tensor",
        ),
        "addr": re.compile(
            r"lda|ldb|ldc|offset|gather|scatter|index",
            re.I,
        ),
        "hint": re.compile(r"hint|prefetch|__launch", re.I),
        "cluster": re.compile(r"cluster|Die\b|dieId|get_thread_idx", re.I),
        "compute": re.compile(
            r"TMATMUL(?:_ACC)?|TADD|TMUL|TCVT|TGELU|TEXP|TROWSUM|TMAX",
        ),
        "pipe_switch": re.compile(
            r"TLOAD(?:_CUBE)?|TSTORE(?:_CUBE)?|TMATMUL|TCVT",
        ),
        "scale": re.compile(r"scale|mxscale|dequant|quant", re.I),
        "api_decl": re.compile(
            r"^\s*(?:template\s*<[^>]*>\s*)?(?:__attribute__\(\(noinline\)\)\s*)?"
            r"(?:void|auto)\s+(\w+)\s*\(",
            re.M,
        ),
    },
    "SIMT": {
        "dma": re.compile(
            r"cudaMemcpy(?:Async)?|cp\.async|TMA\b|__ldg|ld\.global|st\.global|"
            r"memcpy_async",
            re.I,
        ),
        "dma_call": re.compile(
            r"cudaMemcpy(?:Async)?\s*\(|cp\.async|memcpy_async",
            re.I,
        ),
        "coh": re.compile(
            r"__threadfence(?:_system|_block)?|cudaDeviceSynchronize|"
            r"__ldcg|__ldca|__ldcs|cache::",
        ),
        "sync_inter": re.compile(
            r"cooperative_groups|grid\.sync|this_grid|nvshmem|nccl|"
            r"this_cluster",
            re.I,
        ),
        "sync_intra": re.compile(
            r"__syncthreads|__syncwarp|barrier\.sync|atomicAdd|atomicExch|"
            r"atomicCAS",
        ),
        "pc": re.compile(
            r"producer|consumer|pipeline|cuda::pipeline|memcpy_async",
            re.I,
        ),
        "scalar": re.compile(
            r"threadIdx|blockIdx|blockDim|gridDim|warpSize|for\s*\(",
        ),
        "alloc": re.compile(
            r"cudaMalloc|cudaFree|__shared__|extern\s+__shared__",
        ),
        "addr": re.compile(
            r"gather|scatter|stride|offset|\bidx\s*=",
            re.I,
        ),
        "hint": re.compile(
            r"__launch_bounds__|__restrict__|__ldg|__prefetch|"
            r"__builtin_assume|__forceinline__",
        ),
        "cluster": re.compile(
            r"clusterDim|clusterIdx|this_cluster|sm_90|cg::cluster",
            re.I,
        ),
        "compute": re.compile(
            r"__hmma|wmma|mma\.sync|cublas|cudnn|fmaf|expf|sinf|__expf",
        ),
        "pipe_switch": re.compile(
            r"__syncthreads|cp\.async|cuda::pipeline|bar\.sync",
        ),
        "scale": re.compile(
            r"scale|per.?channel|per.?block|dequant|antiquant",
            re.I,
        ),
        "api_decl": re.compile(
            r"__(?:global|device|host)__\s+(?:inline\s+)?[\w:<>,\s\*&]+?\s+(\w+)\s*\(",
        ),
    },
}

CONTROL_FLOW = re.compile(
    r"\b(?:if|else\s+if|for|while|case|catch)\b|\?|&&|\|\|"
)
COMMENT_LINE = re.compile(r"^\s*(?://|/\*|\*|\#(?:if|endif|define|include)\b)")
BLANK = re.compile(r"^\s*$")
ENUM_USE = re.compile(r"\b([A-Z][A-Z0-9_]+)\b")


def load_operators(path: Optional[os.PathLike] = None) -> dict:
    p = Path(path) if path else DEFAULT_OPERATORS
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def arch_file_ok(path: Path, arch: str) -> bool:
    name = path.name.lower()
    suf = path.suffix.lower()
    if arch != "SIMT" and suf in {".cu", ".cuh"}:
        return False
    if arch != "AscendC" and name.startswith("ascendc"):
        return False
    if arch != "SuperScalar" and ("superscalar" in name or "supernpu" in name):
        return False
    if arch != "SIMT" and name.startswith("simt"):
        return False
    return True


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith(".")


def iter_source_files(root: Path, extensions: Sequence[str]) -> Iterable[Path]:
    if not root.exists():
        return
    ext_set = {e.lower() for e in extensions}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in ext_set:
                yield p


def _rel_parts(path: Path, root: Path) -> tuple:
    try:
        return path.resolve().relative_to(root.resolve()).parts
    except (ValueError, OSError):
        return path.parts


def _under_skipped(path: Path, root: Path) -> bool:
    return any(_should_skip_dir(p) for p in _rel_parts(path, root))


def match_globs(root: Path, globs: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for g in globs:
        try:
            matches = list(root.glob(g))
        except ValueError:
            continue
        for p in matches:
            if _under_skipped(p, root):
                continue
            if p.is_file() and p not in seen:
                seen.add(p)
                found.append(p)
            elif p.is_dir():
                for f in iter_source_files(p, SOURCE_EXTS):
                    if f not in seen and not _under_skipped(f, root):
                        seen.add(f)
                        found.append(f)
    return found


def _fold_token(s: str) -> str:
    """Normalize path/token so arg_max matches argmax, drop_out matches dropout."""
    return s.lower().replace("_", "").replace("-", "")


def match_by_tokens(root: Path, tokens: Sequence[str], extensions: Sequence[str]) -> List[Path]:
    """Fallback: any source file whose path contains one of the tokens."""
    toks = [_fold_token(t) for t in tokens if t and t.lower() not in ("na", "类")]
    toks = [t for t in toks if t]
    if not toks:
        return []
    hit = []
    for f in iter_source_files(root, extensions):
        s = _fold_token(str(f))
        if any(t in s for t in toks):
            hit.append(f)
    return hit


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def code_lines(text: str) -> List[str]:
    cleaned = strip_comments(text)
    return [ln for ln in cleaned.splitlines() if not BLANK.match(ln)]


def cyclomatic(lines: Sequence[str]) -> int:
    v = 1
    for ln in lines:
        v += len(CONTROL_FLOW.findall(ln))
    return v


def first_dma_index(lines: Sequence[str], dma_re: re.Pattern) -> int:
    for i, ln in enumerate(lines):
        if dma_re.search(ln):
            return i
    return len(lines)


def analyze_text(text: str, arch: str) -> dict:
    pats = ARCH_PATTERNS[arch]
    lines = code_lines(text)
    total = len(lines)

    def count(key: str) -> int:
        rgx = pats[key]
        return sum(1 for ln in lines if rgx.search(ln))

    dma = count("dma")
    compute = count("compute")
    head_idx = first_dma_index(lines, pats["dma_call"])
    api_names = pats["api_decl"].findall(text)
    enums = set()
    for ln in lines:
        enums.update(ENUM_USE.findall(ln))
    hint_kinds = set()
    if pats["hint"].search(text):
        for m in pats["hint"].finditer(text):
            hint_kinds.add(m.group(0).lower())

    return {
        "L_total": total,
        "L_dma": dma,
        "L_dma_args": dma,
        "L_coh": count("coh"),
        "L_sync": count("sync_inter"),
        "L_intra_sync": count("sync_intra"),
        "L_pc": count("pc"),
        "L_scalar": count("scalar"),
        "L_alloc": count("alloc"),
        "L_addr": count("addr"),
        "L_compute": compute,
        "L_scale": count("scale"),
        "L_head": head_idx,
        "P_src": count("pipe_switch"),
        "N_alive": max(1, count("alloc")),
        "V_base": cyclomatic(lines),
        "f_cluster": 1 if count("cluster") > 0 else 0,
        "N_hint_used": len(hint_kinds),
        "api_names": sorted(set(api_names)),
        "N_api": len(set(api_names)),
        "N_enum_used": len(enums),
        "files": 1,
    }


def _zero_metrics() -> dict:
    return {
        "L_total": 0,
        "L_dma": 0,
        "L_dma_args": 0,
        "L_coh": 0,
        "L_sync": 0,
        "L_intra_sync": 0,
        "L_pc": 0,
        "L_scalar": 0,
        "L_alloc": 0,
        "L_addr": 0,
        "L_compute": 0,
        "L_scale": 0,
        "L_head": 0,
        "P_src": 0,
        "N_alive": 0,
        "V_base": 1,
        "f_cluster": 0,
        "N_hint_used": 0,
        "api_names": [],
        "N_api": 0,
        "N_enum_used": 0,
        "files": 0,
        "file_list": [],
        "missing": True,
    }


def merge_metrics(parts: List[dict]) -> dict:
    if not parts:
        return _zero_metrics()
    out = _zero_metrics()
    out["missing"] = False
    names: List[str] = []
    files: List[str] = []
    cluster = 0
    hints = 0
    enums = 0
    for p in parts:
        for k, v in p.items():
            if k in ("api_names", "file_list", "missing"):
                continue
            if k == "N_alive":
                out[k] = max(out[k], v)
            elif k in ("V_base",):
                out[k] += v
            elif k == "f_cluster":
                cluster = max(cluster, v)
            elif k == "N_hint_used":
                hints += v
            elif k == "N_enum_used":
                enums = max(enums, v)
            elif isinstance(v, (int, float)):
                out[k] += v
        names.extend(p.get("api_names") or [])
        files.extend(p.get("file_list") or [])
    out["api_names"] = sorted(set(names))
    out["N_api"] = len(out["api_names"])
    out["f_cluster"] = cluster
    out["N_hint_used"] = hints
    out["N_enum_used"] = enums
    out["file_list"] = files
    out["V_base"] = max(1, out["V_base"])
    return out


def analyze_files(files: Sequence[Path], arch: str, root: Optional[Path] = None) -> dict:
    parts = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = analyze_text(text, arch)
        rel = str(fp.relative_to(root)) if root else str(fp)
        m["file_list"] = [rel]
        parts.append(m)
    return merge_metrics(parts)


def extract_arch(root: Path, arch: str, operators: dict) -> dict:
    cfg = operators["source_roots"][arch]
    exts = cfg.get("extensions") or list(SOURCE_EXTS)
    result = {"arch": arch, "root": str(root), "operators": {}}
    for op in operators["operators"]:
        globs = op["globs"].get(arch) or []
        files = match_globs(root, globs)
        if not files:
            tokens = []
            for part in re.split(r"[,/·\s]+", str(op.get("op") or "")):
                if part:
                    tokens.append(part)
            # Pattern title before "·" is a generic category (Sparse, Norm, …)
            # and must not be used as a path token.
            pat = str(op.get("pattern") or "")
            specific = pat.split("·", 1)[1] if "·" in pat else ""
            for part in re.split(r"[,/·\s]+", specific):
                if part and len(part) >= 3:
                    tokens.append(part)
            files = match_by_tokens(root, tokens, exts)
        # Prefer op_kernel / solution subtrees when mixed hits exist.
        prefer = set(cfg.get("prefer_subdirs") or [])
        if prefer and files:
            preferred = [f for f in files if any(p in f.parts for p in prefer)]
            if preferred:
                files = preferred
        files = [f for f in files if f.suffix.lower() in {e.lower() for e in exts} and arch_file_ok(f, arch)]
        metrics = analyze_files(files, arch, root)
        result["operators"][str(op["id"])] = {
            "id": op["id"],
            "pattern": op["pattern"],
            "op": op["op"],
            "weight": op["weight"],
            "metrics": metrics,
        }
    return result


def extract_all(roots: Dict[str, str], operators_path: Optional[os.PathLike] = None) -> dict:
    from local_ops import expand_local_path  # local Windows/Linux path helper

    operators = load_operators(operators_path)
    out = {
        "version": 1,
        "kind": "source_extract",
        "arches": {},
    }
    for arch, path in roots.items():
        if not path:
            continue
        root = expand_local_path(str(path))
        if root is None:
            root = Path(str(path).replace("\\", "/")).expanduser()
        if root.exists():
            root = root.resolve()
        out["arches"][arch] = extract_arch(root, arch, operators)
    return out


def extract_from_named_sources(
    files_by_arch_op: Dict[str, Dict[str, List[Tuple[str, str]]]],
    operators_path: Optional[os.PathLike] = None,
) -> dict:
    """files_by_arch_op[arch][op_id] = [(filename, text), ...] for browser uploads."""
    operators = load_operators(operators_path)
    op_meta = {str(o["id"]): o for o in operators["operators"]}
    out = {"version": 1, "kind": "source_extract", "arches": {}}
    for arch, by_op in files_by_arch_op.items():
        arch_out = {"arch": arch, "root": "(upload)", "operators": {}}
        for op in operators["operators"]:
            oid = str(op["id"])
            parts = []
            for name, text in by_op.get(oid, []):
                m = analyze_text(text, arch)
                m["file_list"] = [name]
                parts.append(m)
            metrics = merge_metrics(parts)
            meta = op_meta[oid]
            arch_out["operators"][oid] = {
                "id": meta["id"],
                "pattern": meta["pattern"],
                "op": meta["op"],
                "weight": meta["weight"],
                "metrics": metrics,
            }
        out["arches"][arch] = arch_out
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract source metrics from benchmark operators")
    ap.add_argument("--ascendc", default="", help="Root of ops-nn/ops-math/ops-transformer checkout")
    ap.add_argument("--superscalar", default="", help="SuperNpuBench kernels root (solution/)")
    ap.add_argument("--simt", default="", help="CUDA kernel root")
    ap.add_argument("--operators", default=str(DEFAULT_OPERATORS))
    ap.add_argument("--out", default="extract_result.json")
    args = ap.parse_args()
    roots = {}
    if args.ascendc:
        roots["AscendC"] = args.ascendc
    if args.superscalar:
        roots["SuperScalar"] = args.superscalar
    if args.simt:
        roots["SIMT"] = args.simt
    if not roots:
        raise SystemExit("Provide at least one of --ascendc / --superscalar / --simt")
    data = extract_all(roots, args.operators)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
