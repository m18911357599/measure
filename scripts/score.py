#!/usr/bin/env python3
"""Score every x.y.z usability sub-item from source extract + HW spec + benchmark perf.

Formulas follow measure.md §3. Scores are in [0, 10].
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence

ARCHES = ["AscendC", "SuperScalar", "SIMT"]
TIERS = ["60%", "90%", "99%"]
DIFF_W = {"易": 1, "中": 3, "难": 5}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def clamp10(x: float) -> float:
    return max(0.0, min(10.0, x))


def ln_score(val: float, vmin: float, vmax: float) -> float:
    """Smaller is better, log interpolation between vmin (10) and vmax (0)."""
    if val <= 0:
        return 10.0
    if val <= vmin:
        return 10.0
    if val >= vmax:
        return 0.0
    return 10.0 * (math.log(vmax) - math.log(val)) / (math.log(vmax) - math.log(vmin))


def cost_score(x: float, xref: float) -> float:
    if xref <= 0:
        return 10.0
    return 10.0 * clamp(1.0 - x / xref)


def avg(xs: Sequence[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def weighted_avg(pairs: Sequence[tuple]) -> Optional[float]:
    """pairs: (weight, score)"""
    num = den = 0.0
    for w, s in pairs:
        if s is None or w is None:
            continue
        num += w * s
        den += w
    return num / den if den else None


def _m(extract: dict, arch: str, oid: str) -> Optional[dict]:
    try:
        return extract["arches"][arch]["operators"][str(oid)]["metrics"]
    except (KeyError, TypeError):
        return None


def _ops(extract: dict, arch: str) -> List[dict]:
    arch_data = (extract or {}).get("arches", {}).get(arch, {})
    ops = arch_data.get("operators") or {}
    return [ops[k] for k in sorted(ops, key=lambda x: int(x))]


def _present(metrics: Optional[dict]) -> bool:
    return bool(metrics) and not metrics.get("missing") and metrics.get("L_total", 0) > 0


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _perf(perf: dict, arch: str, oid: str) -> dict:
    return (((perf or {}).get(arch) or {}).get(str(oid)) or {})


def _tier(perf_op: dict, tier: str) -> dict:
    return (perf_op.get("tiers") or {}).get(tier) or {}


def _hw(hw: dict, arch: str) -> dict:
    if not hw:
        return {}
    raw = hw[arch] if arch in hw and isinstance(hw[arch], dict) else hw
    out = {}
    for k, v in raw.items():
        if v == "" or v is None:
            continue
        out[k] = v
    if "buffers" in raw and isinstance(raw["buffers"], list):
        bufs = []
        for b in raw["buffers"]:
            if not isinstance(b, dict):
                continue
            bb = {bk: bv for bk, bv in b.items() if bv != "" and bv is not None}
            if bb.get("name") or "align_B" in bb or "cap_KiB" in bb:
                bufs.append(bb)
        out["buffers"] = bufs
    return out


def _pat_w(patterns: List[dict], oid: int, cat: str) -> float:
    for p in patterns or []:
        if int(p.get("id", 0)) == int(oid):
            diff = (p.get("difficulty") or {}).get(cat)
            return float(DIFF_W.get(diff, 1))
    return 1.0


# ---------------------------------------------------------------------------
# Per-operator scorers (return None if data missing)
# ---------------------------------------------------------------------------

def score_1_a_1_op(hw: dict, perf_op: dict) -> Optional[float]:
    N = float(hw.get("N") or 64)
    dims = []
    for dim, vmin, vmax, key in (
        ("TileSize", 8192, 65536, "tileSize"),
        ("BurstLen", 32, 256, "burstLen"),
        ("stride", 1, 64, "stride"),
        ("coreConc", 1, max(N, 1), "coreConc"),
    ):
        ss = []
        for tier in TIERS:
            val = _tier(perf_op, tier).get(key)
            if val is None:
                continue
            if dim == "coreConc":
                ss.append(ln_score(float(val), 1, N))
            else:
                ss.append(ln_score(float(val), vmin, vmax))
        if ss:
            dims.append(sum(ss) / len(ss))
    return avg(dims)


def score_3_a_1(hw: dict) -> Optional[float]:
    bufs = hw.get("buffers") or []
    scores = []
    for b in bufs:
        if "align_B" not in b:
            continue
        s = float(b.get("align_B") or 0)
        if s <= 0:
            scores.append(10.0)
        else:
            scores.append(ln_score(s, 1, 64))
    if not scores:
        return None
    return avg(scores)


def score_3_a_2_op(metrics: dict, perf_op: dict) -> Optional[float]:
    p0 = perf_op.get("p0")
    if p0 is not None and float(p0) >= 0.95:
        return 10.0
    Lt = max(1, int(metrics.get("L_total") or 1))
    scores = []
    for tier in TIERS:
        t = _tier(perf_op, tier)
        Lc = t.get("Lc")
        dV = t.get("dV")
        if Lc is None and dV is None:
            # default implementation already good enough if p0 >= tier
            continue
        Lc = float(Lc or 0)
        dV = float(dV or 1)
        Sg = 10.0 * (math.log(Lt) - math.log(Lc + 1)) / math.log(Lt) if Lt > 1 else 10.0
        Sg = clamp10(Sg)
        Sv = 10.0 * clamp(2 - math.exp(dV - 1), 0, 1)
        scores.append(0.6 * Sg + 0.4 * Sv)
    return avg(scores)


def score_3_a_3(hw: dict) -> Optional[float]:
    pref = float(hw.get("pipe_ref") or 3)
    keys = ["P_gm2vec", "P_gm2cube", "P_cube2vec", "P_vec2cube"]
    if not any(k in hw for k in keys):
        return None
    ss = []
    for k in keys:
        p = float(hw.get(k) or 0)
        ss.append(10.0 * clamp(1 - p / pref))
    return avg(ss)


def score_3_a_4_op(metrics: dict, hw: dict, perf_op: dict) -> Optional[float]:
    n_alive = float(metrics.get("N_alive") or 1)
    bufs = hw.get("buffers") or []
    if not bufs:
        return None
    scores = []
    for tier in TIERS:
        T = _tier(perf_op, tier).get("tileSize")
        if T is None:
            continue
        T = float(T)
        buf_s = []
        for b in bufs:
            cap = float(b.get("cap_KiB") or 0) * 1024.0
            if cap <= 0:
                continue
            rho = n_alive * T / cap
            buf_s.append(10.0 if rho <= 1 else min(10.0, 10.0 / rho))
        if buf_s:
            scores.append(avg(buf_s))
    return avg(scores)


def score_ratio_lines(num: float, den: float, xref: float) -> float:
    r = _ratio(num, den)
    return cost_score(r, xref)


def score_3_a_5_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_dma"], m["L_total"], 0.3)


def score_3_b_1_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_coh"], m["L_total"], 0.2)


def score_3_b_2(hw: dict, perf: dict, extract: dict, arch: str) -> Optional[float]:
    n_hint = hw.get("N_hint")
    if n_hint is None:
        # fall back to max used in source
        used = []
        for op in _ops(extract, arch):
            if _present(op.get("metrics")):
                used.append(op["metrics"].get("N_hint_used") or 0)
        n_hint = max(used) if used else None
    if n_hint is None:
        return None
    S_hint = clamp10(10 - 2 * float(n_hint))
    gaps = []
    for op in _ops(extract, arch):
        g = _perf(perf, arch, op["id"]).get("hint_gap")
        if g is not None:
            gaps.append(float(g))
    if not gaps:
        S_gap = None
    else:
        gap = sum(gaps) / len(gaps)
        if gap < 0.05:
            S_gap = 10.0
        elif gap < 0.15:
            S_gap = 8.0
        elif gap < 0.30:
            S_gap = 5.0
        else:
            S_gap = 3.0
    if S_gap is None:
        return S_hint
    return 0.45 * S_hint + 0.55 * S_gap


def score_3_b_3_ops(extract: dict, arch: str, patterns: list, cat: str) -> Optional[float]:
    pairs = []
    for op in _ops(extract, arch):
        m = op.get("metrics")
        if not _present(m):
            continue
        s = clamp10(10 - 5 * float(m.get("f_cluster") or 0))
        pairs.append((_pat_w(patterns, op["id"], cat) * op.get("weight", 1), s))
    return weighted_avg(pairs)


def score_3_b_4(hw: dict, perf: dict, extract: dict, arch: str) -> Optional[float]:
    ic = float(hw.get("ICache_KiB") or 0)
    dc = float(hw.get("DCache_KiB") or 0)
    bin_kib = hw.get("binary_stack_KiB")
    S_ratio = None
    if ic + dc > 0 and bin_kib is not None:
        r = float(bin_kib) / (ic + dc)
        # r<=1 → 10; every +10% over → -1
        S_ratio = 10.0 * clamp(1 - max(r - 1, 0) / 0.1)
    misses = []
    for op in _ops(extract, arch):
        m = _perf(perf, arch, op["id"]).get("icache_miss")
        if m is not None:
            misses.append(float(m))
    S_miss = None
    if misses:
        miss = sum(misses) / len(misses)
        if miss < 0.01:
            S_miss = 10.0
        elif miss < 0.05:
            S_miss = 8.0
        elif miss < 0.15:
            S_miss = 5.0
        else:
            S_miss = 3.0
    if S_ratio is None and S_miss is None:
        return None
    if S_ratio is None:
        return S_miss
    if S_miss is None:
        return S_ratio
    return 0.40 * S_ratio + 0.60 * S_miss


def score_4_a_1(hw: dict, extract: dict, arch: str) -> Optional[float]:
    n_api = hw.get("N_api")
    n_viol = hw.get("N_viol")
    if n_api is not None and n_viol is not None and float(n_api) > 0:
        return 10.0 * (1 - float(n_viol) / float(n_api))
    return None


def score_4_a_2(hw: dict) -> Optional[float]:
    n_hw = hw.get("N_hw")
    n_dup = hw.get("N_api_dup")
    if n_hw is None or n_dup is None or float(n_hw) <= 0:
        return None
    eta = float(n_dup) / float(n_hw)
    S_eta = 10.0 * (1 - eta)
    r = hw.get("eta_ratio_vs_baseline")
    S_comp = 10.0 * clamp(1 / r, 0, 1) if r else S_eta
    return 0.65 * S_eta + 0.35 * S_comp


def score_4_a_3(hw: dict, extract: dict, arch: str) -> Optional[float]:
    pi = hw.get("optional_param_ratio")
    S_pi = None
    if pi is not None:
        S_pi = 10.0 * clamp(1 - float(pi) / 0.5)
    n_def = hw.get("N_enum_def")
    S_eps = None
    if n_def:
        used = []
        for op in _ops(extract, arch):
            m = op.get("metrics")
            if _present(m):
                used.append(m.get("N_enum_used") or 0)
        if used:
            eps = min(1.0, max(used) / float(n_def))
            S_eps = 10.0 * clamp((eps - 0.3) / 0.7)
    if S_pi is None and S_eps is None:
        return None
    if S_pi is None:
        return S_eps
    if S_eps is None:
        return S_pi
    return 0.5 * S_pi + 0.5 * S_eps


def score_4_a_4(hw: dict) -> Optional[float]:
    keys = ["r_hw", "r_simt", "r_sw", "r_cover"]
    if not any(k in hw for k in keys):
        return None
    r_hw = float(hw.get("r_hw") or 0)
    r_simt = float(hw.get("r_simt") or 0)
    r_sw = float(hw.get("r_sw") or 0)
    r_cover = float(hw.get("r_cover") or 0)
    S_sw = 10.0 * (1 - r_sw)  # more software emulation → worse
    return 0.25 * 10 * r_hw + 0.25 * 10 * r_simt + 0.15 * S_sw + 0.35 * 10 * r_cover


def score_4_b_1_op(perf_op: dict) -> Optional[float]:
    fused = perf_op.get("T_fused")
    raw = perf_op.get("T_no_fused")
    if fused is None or raw is None or float(raw) == 0:
        return None
    rho = float(fused) / float(raw)
    return 10.0 * clamp(1 - abs(rho - 1) / 0.25)


def score_4_b_2(hw: dict, extract: dict, arch: str, perf: dict, patterns: list) -> Optional[float]:
    r_mma = hw.get("r_mma")
    ulp = hw.get("ulp")
    c_conv = hw.get("c_conv")
    rtn = hw.get("round_rtn", 0)
    sr = hw.get("round_sr", 0)
    r_ieee = hw.get("r_ieee")
    S_mma = 10.0 * clamp(float(r_mma)) if r_mma is not None else None
    S_ulp = None
    if ulp is not None:
        S_ulp = 10.0 * clamp(1 - max(float(ulp) - 2.0, 0) / 2.0)
    S_conv = None
    if c_conv is not None:
        S_conv = 10.0 / (2 ** float(c_conv)) + 0.5 * float(rtn or 0) + 0.5 * float(sr or 0)
        S_conv = clamp10(S_conv)
    scale_pairs = []
    mfu_pairs = []
    for op in _ops(extract, arch):
        m = op.get("metrics")
        w = op.get("weight", 1)
        if _present(m) and m["L_total"] > 0:
            g = m["L_scale"] / m["L_total"]
            scale_pairs.append((w, 10.0 * clamp(1 - g / 0.15)))
        p = _perf(perf, arch, op["id"])
        if p.get("mfu_quant") is not None and p.get("mfu_fp16"):
            r = float(p["mfu_quant"]) / float(p["mfu_fp16"])
            mfu_pairs.append((w, 10.0 * clamp(r)))
    S_scale = weighted_avg(scale_pairs)
    S_mfu = weighted_avg(mfu_pairs)
    S_ieee = 10.0 * float(r_ieee) if r_ieee is not None else None
    parts = [
        (0.25, S_mma), (0.15, S_ulp), (0.15, S_conv),
        (0.15, S_scale), (0.20, S_mfu), (0.10, S_ieee),
    ]
    present = [(w, s) for w, s in parts if s is not None]
    if not present:
        return None
    # renormalize weights among available components
    tw = sum(w for w, _ in present)
    return sum(w / tw * s for w, s in present)


def score_5_a_1_op(m: dict, perf_op: dict) -> Optional[float]:
    p_src = m.get("P_src") if _present(m) else None
    p_rt = perf_op.get("P_runtime")
    if p_src is None or p_rt is None or float(p_src) <= 0:
        return None
    gamma = float(p_rt) / float(p_src)
    return 10.0 * clamp(2 - gamma, 0, 1)


def score_5_b_1_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_sync"], m["L_total"], 0.3)


def score_5_b_2_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_intra_sync"], m["L_total"], 0.25)


def score_5_c_1_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_pc"], m["L_total"], 0.35)


def score_5_d_1_op(m: dict, perf_op: dict) -> Optional[float]:
    ts = []
    for tier in TIERS:
        t = _tier(perf_op, tier)
        t_s, t_m = t.get("T_scalar"), t.get("T_maxPipe")
        S_g = None
        if t_s is not None and t_m and float(t_m) > 0:
            g = float(t_s) / float(t_m)
            S_g = 10.0 * clamp(1 - (g - 1) / 0.05)
        S_c = None
        if _present(m):
            r = m["L_scalar"] / m["L_total"]
            S_c = 10.0 * clamp(1 - r / 0.3)
        if S_g is None and S_c is None:
            continue
        if S_g is None:
            ts.append(S_c)
        elif S_c is None:
            ts.append(S_g)
        else:
            ts.append(0.55 * S_g + 0.45 * S_c)
    return avg(ts)


def score_5_e_1_op(m: dict, perf_op: dict) -> Optional[float]:
    S_t = None
    th, tk = perf_op.get("T_head"), perf_op.get("T_kernel")
    if th is not None and tk and float(tk) > 0:
        S_t = 10.0 * clamp(1 - float(th) / float(tk) / 0.2)
    S_c = None
    if _present(m) and m["L_total"] > 0:
        S_c = 10.0 * clamp(1 - m["L_head"] / m["L_total"] / 0.25)
    if S_t is None and S_c is None:
        return None
    if S_t is None:
        return S_c
    if S_c is None:
        return S_t
    return 0.55 * S_t + 0.45 * S_c


def score_6_a_1_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_dma_args"], m["L_total"], 0.35)


def score_6_a_2_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_addr"], m["L_total"], 0.3)


def score_6_b_1_op(m: dict) -> Optional[float]:
    if not _present(m):
        return None
    return score_ratio_lines(m["L_alloc"], m["L_total"], 0.25)


def _hw_rate(hw: dict, num_key: str, den_key: str, den_default: Optional[float] = None) -> Optional[float]:
    n = hw.get(num_key)
    d = hw.get(den_key, den_default)
    if n is None or d is None or float(d) <= 0:
        return None
    return 10.0 * clamp(float(n) / float(d))


def score_7_a_1(hw: dict) -> Optional[float]:
    return _hw_rate(hw, "exc_detected", "exc_defined")


def score_7_a_2(hw: dict) -> Optional[float]:
    p_det = hw.get("p_det")
    if p_det is None:
        return None
    S_det = 10.0 * float(p_det)
    fn = float(hw.get("p_fn") or 0)
    fp = float(hw.get("p_fp") or 0)
    dup = float(hw.get("p_dup") or 0)
    sdc = float(hw.get("p_sdc") or 0)
    S_noise = 10.0 * (1 - (fn + fp + dup + sdc) / 4)
    return 0.60 * S_det + 0.40 * clamp10(S_noise)


def score_7_a_3(hw: dict) -> Optional[float]:
    c_acc = hw.get("c_acc")
    c_comp = hw.get("c_comp")
    if c_acc is None and c_comp is None:
        return None
    if c_acc is None:
        return 10.0 * float(c_comp)
    if c_comp is None:
        return 10.0 * float(c_acc)
    return 0.55 * 10 * float(c_acc) + 0.45 * 10 * float(c_comp)


def score_7_a_4(hw: dict) -> Optional[float]:
    return _hw_rate(hw, "root_ok", "root_total") or (
        10.0 * float(hw["r_root"]) if hw.get("r_root") is not None else None
    )


def score_7_b_1(hw: dict) -> Optional[float]:
    n = hw.get("bp_supported")
    if n is None:
        return None
    return 10.0 * clamp(float(n) / 12.0)


def score_7_b_2(hw: dict) -> Optional[float]:
    return _hw_rate(hw, "bp_loc_ok", "bp_loc_total") or (
        10.0 * float(hw["r_loc"]) if hw.get("r_loc") is not None else None
    )


def score_7_b_3(hw: dict) -> Optional[float]:
    return _hw_rate(hw, "pc_ok", "pc_total") or (
        10.0 * float(hw["r_pc"]) if hw.get("r_pc") is not None else None
    )


def score_7_b_4(hw: dict) -> Optional[float]:
    return _hw_rate(hw, "halt_ok", "halt_total") or (
        10.0 * float(hw["r_halt"]) if hw.get("r_halt") is not None else None
    )


def score_7_c_1(hw: dict) -> Optional[float]:
    return _hw_rate(hw, "obs_ok", "obs_total") or (
        10.0 * float(hw["r_obscov"]) if hw.get("r_obscov") is not None else None
    )


def score_7_d_1(hw: dict) -> Optional[float]:
    return _hw_rate(hw, "evt_ok", "evt_total") or (
        10.0 * float(hw["r_evtcov"]) if hw.get("r_evtcov") is not None else None
    )


def score_7_d_2(hw: dict) -> Optional[float]:
    e = hw.get("e_rel")
    if e is None:
        return None
    return 10.0 * clamp(1 - float(e) / 0.1)


def score_7_d_3(hw: dict) -> Optional[float]:
    r = hw.get("r_loss")
    if r is None:
        return None
    return 10.0 * clamp((float(r) - 0.95) / 0.05)


def score_7_d_4(hw: dict) -> Optional[float]:
    n = hw.get("bn_ok")
    if n is None:
        return None
    return 10.0 * clamp(float(n) / 9.0)


def score_7_d_5(hw: dict) -> Optional[float]:
    return 10.0 * float(hw["r_expl"]) if hw.get("r_expl") is not None else None


def score_7_d_6(hw: dict) -> Optional[float]:
    dt = hw.get("delta_t")
    db = hw.get("delta_b")
    if dt is None and db is None:
        return None
    S_t = 10.0 * clamp(1 - float(dt) / 0.05) if dt is not None else None
    S_p = 10.0 * clamp(1 - float(db) / 0.02) if db is not None else None
    if S_t is None:
        return S_p
    if S_p is None:
        return S_t
    return 0.55 * S_t + 0.45 * S_p


def score_7_d_7(hw: dict) -> Optional[float]:
    return 10.0 * float(hw["r_tasklink"]) if hw.get("r_tasklink") is not None else None


def score_8_a_1_op(perf_op: dict) -> Optional[float]:
    pa, pt = perf_op.get("P_agent"), perf_op.get("P_theory")
    if pa is None or pt is None or float(pt) <= 0:
        return None
    return 10.0 * clamp(float(pa) / float(pt))


def score_9_a_1_op(perf_op: dict) -> Optional[float]:
    vers = perf_op.get("versions") or []
    n = perf_op.get("N_versions")
    if n is None:
        n = len(vers) if vers else None
    if n is None:
        return None
    n = float(n)
    S_n = 10.0 * clamp(1 - (n - 1) / 5)
    gammas, rhos = [], []
    for i in range(1, len(vers)):
        prev, cur = vers[i - 1], vers[i]
        L0 = float(prev.get("lines") or 0)
        dL = float(cur.get("delta_lines") if cur.get("delta_lines") is not None else abs((cur.get("lines") or 0) - L0))
        if L0 > 0:
            gammas.append(dL / L0)
        t0, t1 = prev.get("time"), cur.get("time")
        if t0 and t1:
            rhos.append(float(t1) / float(t0))
    S_g = 10.0 * clamp(1 - (sum(gammas) / len(gammas)) / 0.3) if gammas else None
    S_r = 10.0 * clamp((1 - sum(rhos) / len(rhos)) / 0.3) if rhos else None
    parts = [(0.25, S_n), (0.35, S_g), (0.40, S_r)]
    present = [(w, s) for w, s in parts if s is not None]
    tw = sum(w for w, _ in present)
    return sum(w / tw * s for w, s in present) if present else None


def _agg_ops(extract, arch, perf, patterns, cat, fn) -> Optional[float]:
    pairs = []
    for op in _ops(extract, arch):
        m = op.get("metrics") or {}
        p = _perf(perf, arch, op["id"])
        s = fn(m, p)
        if s is None:
            continue
        w = float(op.get("weight") or 1) * _pat_w(patterns, op["id"], cat)
        pairs.append((w, s))
    return weighted_avg(pairs)


def score_arch(
    arch: str,
    extract: dict,
    hw_all: dict,
    perf: dict,
    patterns: Optional[List[dict]] = None,
) -> Dict[str, Optional[float]]:
    hw = _hw(hw_all, arch)
    patterns = patterns or []
    out: Dict[str, Optional[float]] = {}

    out["1.a.1"] = _agg_ops(
        extract, arch, perf, patterns, "系统规格",
        lambda m, p: score_1_a_1_op(hw, p),
    )
    out["3.a.1"] = score_3_a_1(hw)
    out["3.a.2"] = _agg_ops(extract, arch, perf, patterns, "内存模型", score_3_a_2_op)
    out["3.a.3"] = score_3_a_3(hw)
    out["3.a.4"] = _agg_ops(
        extract, arch, perf, patterns, "内存模型",
        lambda m, p: score_3_a_4_op(m, hw, p),
    )
    out["3.a.5"] = _agg_ops(extract, arch, perf, patterns, "内存模型", lambda m, p: score_3_a_5_op(m))
    out["3.b.1"] = _agg_ops(extract, arch, perf, patterns, "内存模型", lambda m, p: score_3_b_1_op(m))
    out["3.b.2"] = score_3_b_2(hw, perf, extract, arch)
    out["3.b.3"] = score_3_b_3_ops(extract, arch, patterns, "内存模型")
    out["3.b.4"] = score_3_b_4(hw, perf, extract, arch)
    out["4.a.1"] = score_4_a_1(hw, extract, arch)
    out["4.a.2"] = score_4_a_2(hw)
    out["4.a.3"] = score_4_a_3(hw, extract, arch)
    out["4.a.4"] = score_4_a_4(hw)
    out["4.b.1"] = _agg_ops(extract, arch, perf, patterns, "计算模型", lambda m, p: score_4_b_1_op(p))
    out["4.b.2"] = score_4_b_2(hw, extract, arch, perf, patterns)
    out["5.a.1"] = _agg_ops(extract, arch, perf, patterns, "控制模型", score_5_a_1_op)
    out["5.b.1"] = _agg_ops(extract, arch, perf, patterns, "控制模型", lambda m, p: score_5_b_1_op(m))
    out["5.b.2"] = _agg_ops(extract, arch, perf, patterns, "控制模型", lambda m, p: score_5_b_2_op(m))
    out["5.c.1"] = _agg_ops(extract, arch, perf, patterns, "控制模型", lambda m, p: score_5_c_1_op(m))
    out["5.d.1"] = _agg_ops(extract, arch, perf, patterns, "控制模型", score_5_d_1_op)
    out["5.e.1"] = _agg_ops(extract, arch, perf, patterns, "控制模型", score_5_e_1_op)
    out["6.a.1"] = _agg_ops(extract, arch, perf, patterns, "访存模型", lambda m, p: score_6_a_1_op(m))
    out["6.a.2"] = _agg_ops(extract, arch, perf, patterns, "访存模型", lambda m, p: score_6_a_2_op(m))
    out["6.b.1"] = _agg_ops(extract, arch, perf, patterns, "访存模型", lambda m, p: score_6_b_1_op(m))
    out["7.a.1"] = score_7_a_1(hw)
    out["7.a.2"] = score_7_a_2(hw)
    out["7.a.3"] = score_7_a_3(hw)
    out["7.a.4"] = score_7_a_4(hw)
    out["7.b.1"] = score_7_b_1(hw)
    out["7.b.2"] = score_7_b_2(hw)
    out["7.b.3"] = score_7_b_3(hw)
    out["7.b.4"] = score_7_b_4(hw)
    out["7.c.1"] = score_7_c_1(hw)
    out["7.d.1"] = score_7_d_1(hw)
    out["7.d.2"] = score_7_d_2(hw)
    out["7.d.3"] = score_7_d_3(hw)
    out["7.d.4"] = score_7_d_4(hw)
    out["7.d.5"] = score_7_d_5(hw)
    out["7.d.6"] = score_7_d_6(hw)
    out["7.d.7"] = score_7_d_7(hw)
    out["8.a.1"] = _agg_ops(extract, arch, perf, patterns, "Agent友好", lambda m, p: score_8_a_1_op(p))
    out["9.a.1"] = _agg_ops(extract, arch, perf, patterns, "开发效率", lambda m, p: score_9_a_1_op(p))
    return out


def score_all(extract: dict, hw: dict, perf: dict, patterns: Optional[List[dict]] = None) -> dict:
    scores = {}
    details = {}
    for arch in ARCHES:
        arch_scores = score_arch(arch, extract, hw, perf, patterns)
        details[arch] = arch_scores
        for code, val in arch_scores.items():
            scores.setdefault(code, {})
            scores[code][arch] = None if val is None else round(val, 2)
    return {"kind": "computed_scores", "scores": scores, "by_arch": details}


def merge_into_measure_data(measure: dict, computed: dict, extract: Optional[dict] = None) -> dict:
    """Write computed scores into measure_data.scores without dropping notes.

    None scores (missing HW/perf) are left unchanged so placeholder / spec
    values are not wiped. Source-extract scores overwrite the matching slots.
    """
    from datetime import datetime, timezone

    out = json.loads(json.dumps(measure))
    src = computed.get("scores") or {}
    dest = out.setdefault("scores", {})
    n = 0
    for code, by_arch in src.items():
        dest.setdefault(code, {})
        for arch, val in by_arch.items():
            if val is None:
                continue
            slot = dest[code].setdefault(arch, {})
            slot["score"] = val
            slot["source"] = "computed"
            n += 1
    meta = out.setdefault("meta", {})
    meta["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["source"] = "scripts/measure.py extract+score (local kernels)"
    coverage = {}
    if extract:
        out["extract"] = {
            "version": extract.get("version"),
            "kind": extract.get("kind"),
            "arches": {},
        }
        for arch, blob in (extract.get("arches") or {}).items():
            ops = blob.get("operators") or {}
            found = sum(1 for o in ops.values() if not (o.get("metrics") or {}).get("missing"))
            coverage[arch] = {"found": found, "total": len(ops), "root": blob.get("root")}
            # Keep per-op metrics but drop bulky file lists in the report JSON.
            slim_ops = {}
            for oid, rec in ops.items():
                m = dict(rec.get("metrics") or {})
                files = m.pop("file_list", None) or []
                m["n_files"] = len(files)
                slim_ops[oid] = {
                    "id": rec.get("id"),
                    "pattern": rec.get("pattern"),
                    "op": rec.get("op"),
                    "weight": rec.get("weight"),
                    "metrics": m,
                }
            out["extract"]["arches"][arch] = {
                "arch": arch,
                "root": blob.get("root"),
                "operators": slim_ops,
            }
    meta["local_measure"] = {
        "coverage": coverage,
        "extract_file": "measure_extract.json",
        "scores_file": "measure_scores.json",
        "filled_slots": n,
    }
    return out
