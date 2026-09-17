/* NPU usability: source extraction + scoring (mirrors scripts/extract_source.py + score.py). */
(function (global) {
  const ARCHES = ["AscendC", "SuperScalar", "SIMT"];
  const TIERS = ["60%", "90%", "99%"];
  const DIFF_W = { "易": 1, "中": 3, "难": 5 };

  const SOURCE_KIND = {
    "1.a.1": "benchmark_perf",
    "3.a.1": "hw_spec", "3.a.2": "mixed", "3.a.3": "hw_spec", "3.a.4": "mixed", "3.a.5": "source",
    "3.b.1": "source", "3.b.2": "mixed", "3.b.3": "source", "3.b.4": "mixed",
    "4.a.1": "mixed", "4.a.2": "hw_spec", "4.a.3": "mixed", "4.a.4": "hw_spec",
    "4.b.1": "benchmark_perf", "4.b.2": "mixed",
    "5.a.1": "mixed", "5.b.1": "source", "5.b.2": "source", "5.c.1": "source",
    "5.d.1": "mixed", "5.e.1": "mixed",
    "6.a.1": "source", "6.a.2": "source", "6.b.1": "source",
    "7.a.1": "hw_spec", "7.a.2": "hw_spec", "7.a.3": "hw_spec", "7.a.4": "hw_spec",
    "7.b.1": "hw_spec", "7.b.2": "hw_spec", "7.b.3": "hw_spec", "7.b.4": "hw_spec",
    "7.c.1": "hw_spec",
    "7.d.1": "hw_spec", "7.d.2": "hw_spec", "7.d.3": "hw_spec", "7.d.4": "hw_spec",
    "7.d.5": "hw_spec", "7.d.6": "hw_spec", "7.d.7": "hw_spec",
    "8.a.1": "benchmark_perf",
    "9.a.1": "mixed"
  };

  const KIND_LABEL = {
    source: "源码提取",
    hw_spec: "硬件规格",
    benchmark_perf: "Benchmark性能",
    mixed: "源码+规格/性能"
  };

  const PAT = {
    AscendC: {
      dma: /DataCopy(?:Pad|2D)?|Copy\s*\(|Nd2Nz|Nz2Nd|SetNdDma|TQue|EnQue|DeQue|MTE2|MTE3|GlobalTensor|GM_ADDR/i,
      dma_call: /DataCopy(?:Pad|2D)?\s*\(|Copy\s*\(/,
      coh: /DCCI|dcci|Invalidate|Prefetch|cacheline|CacheMode|SetCacheMode|dcflush|icache|dcache/i,
      sync_inter: /CrossCore(?:Set|Wait)Flag|SyncAll|Ipc(?:Set|Wait)|Notify|GetBlockNum|GetSubBlockIdx/i,
      sync_intra: /SetFlag|WaitFlag|PipeBarrier|pipe_barrier|event_t|TEventID/i,
      pc: /TQue|Producer|Consumer|EnQue|DeQue|GetTPipePtr|TPipe/i,
      scalar: /GetBlockIdx|GetBlockNum|block_idx|GetUserWorkspace|GetSysWorkSpace|printf|assert|for\s*\(/i,
      alloc: /AllocTensor|FreeTensor|InitBuffer|TBuf|TPipe\s+\w+|GetTQueHead/i,
      addr: /Gather|Scatter|stride|offset|indexOffset|gIndex|blockOffset|dataIndex|startIndex/i,
      hint: /\bhint\b|Prefetch|SetCacheMode|persistent/i,
      cluster: /cluster|GetCluster|dieId|DieId|GetDie/i,
      compute: /\b(?:Add|Mul|Muls|Exp|Ln|Abs|Relu|Div|Sub|Max|Min|Mmad|Axpy|Cast|SoftMax|ReduceSum)\s*\(|CubeCompute/,
      pipe_switch: /SetFlag|WaitFlag|PipeBarrier|EnQue|DeQue/i,
      scale: /scale|per.?channel|per.?block|dequant|antiquant|mxscale/i,
      api_decl: /(?:__aicore__|__global__)?\s*(?:inline\s+)?(?:void|auto)\s+(\w+)\s*\(/g
    },
    SuperScalar: {
      dma: /TLOAD(?:_CUBE)?|TSTORE(?:_CUBE)?|TCOPY(?:IN|OUT)?|TMA\b/,
      dma_call: /TLOAD(?:_CUBE)?\s*\(|TSTORE(?:_CUBE)?\s*\(|TCOPY/,
      coh: /fence|invalidate|cache|dcci/i,
      sync_inter: /get_thread_idx|BARRIER|sync_pe|cluster|multi_pe/i,
      sync_intra: /\bwait\b|barrier|BENCHSTART|BENCHEND|PipeWait/i,
      pc: /producer|consumer|pipeline|stage/i,
      scalar: /\bg[MNK]\b|lda|ldb|ldc|for\s*\(|if\s*\(/,
      alloc: /Tile\s*<|CubeTile|CubeAccumulator|scratch|buffer|global_tensor/,
      addr: /lda|ldb|ldc|offset|gather|scatter|index/i,
      hint: /hint|prefetch|__launch/i,
      cluster: /cluster|Die\b|dieId|get_thread_idx/i,
      compute: /TMATMUL(?:_ACC)?|TADD|TMUL|TCVT|TGELU|TEXP|TROWSUM|TMAX/,
      pipe_switch: /TLOAD(?:_CUBE)?|TSTORE(?:_CUBE)?|TMATMUL|TCVT/,
      scale: /scale|mxscale|dequant|quant/i,
      api_decl: /(?:void|auto)\s+(\w+)\s*\(/g
    },
    SIMT: {
      dma: /cudaMemcpy(?:Async)?|cp\.async|TMA\b|__ldg|ld\.global|st\.global|memcpy_async/i,
      dma_call: /cudaMemcpy(?:Async)?\s*\(|cp\.async|memcpy_async/i,
      coh: /__threadfence(?:_system|_block)?|cudaDeviceSynchronize|__ldcg|__ldca|__ldcs|cache::/,
      sync_inter: /cooperative_groups|grid\.sync|this_grid|nvshmem|nccl|this_cluster/i,
      sync_intra: /__syncthreads|__syncwarp|barrier\.sync|atomicAdd|atomicExch|atomicCAS/,
      pc: /producer|consumer|pipeline|cuda::pipeline|memcpy_async/i,
      scalar: /threadIdx|blockIdx|blockDim|gridDim|warpSize|for\s*\(/,
      alloc: /cudaMalloc|cudaFree|__shared__|extern\s+__shared__/,
      addr: /gather|scatter|stride|offset|\bidx\s*=/i,
      hint: /__launch_bounds__|__restrict__|__ldg|__prefetch|__builtin_assume|__forceinline__/,
      cluster: /clusterDim|clusterIdx|this_cluster|sm_90|cg::cluster/i,
      compute: /__hmma|wmma|mma\.sync|cublas|cudnn|fmaf|expf|sinf|__expf/,
      pipe_switch: /__syncthreads|cp\.async|cuda::pipeline|bar\.sync/,
      scale: /scale|per.?channel|per.?block|dequant|antiquant/i,
      api_decl: /__(?:global|device|host)__\s+(?:inline\s+)?[\w:<>,\s\*&]+?\s+(\w+)\s*\(/g
    }
  };

  const CONTROL = /\b(?:if|else\s+if|for|while|case|catch)\b|\?|&&|\|\|/g;
  const ENUM_USE = /\b([A-Z][A-Z0-9_]+)\b/g;
  const SRC_EXT = /\.(cpp|cc|c|h|hpp|cce|inl|cu|cuh|hip)$/i;

  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function clamp10(x) { return clamp(x, 0, 10); }
  function avg(xs) {
    const v = xs.filter(x => x != null && !isNaN(x));
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }
  function wavg(pairs) {
    let n = 0, d = 0;
    for (const [w, s] of pairs) {
      if (s == null || w == null) continue;
      n += w * s; d += w;
    }
    return d ? n / d : null;
  }
  function lnScore(val, vmin, vmax) {
    if (val <= 0 || val <= vmin) return 10;
    if (val >= vmax) return 0;
    return 10 * (Math.log(vmax) - Math.log(val)) / (Math.log(vmax) - Math.log(vmin));
  }
  function cost(x, xref) { return xref <= 0 ? 10 : 10 * clamp(1 - x / xref, 0, 1); }
  function ratio(n, d) { return d > 0 ? n / d : 0; }

  function stripComments(text) {
    return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
  }
  function codeLines(text) {
    return stripComments(text).split(/\r?\n/).filter(ln => ln.trim().length);
  }
  function countRe(lines, re) {
    return lines.reduce((n, ln) => n + (re.test(ln) ? 1 : 0), 0);
  }
  function firstDma(lines, re) {
    for (let i = 0; i < lines.length; i++) if (re.test(lines[i])) return i;
    return lines.length;
  }

  function analyzeText(text, arch) {
    const p = PAT[arch];
    const lines = codeLines(text);
    const total = lines.length;
    const dma = countRe(lines, p.dma);
    const names = [];
    let m;
    const apiRe = new RegExp(p.api_decl.source, "g");
    while ((m = apiRe.exec(text))) names.push(m[1]);
    const enums = new Set();
    lines.forEach(ln => { let e; ENUM_USE.lastIndex = 0; while ((e = ENUM_USE.exec(ln))) enums.add(e[1]); });
    const hints = new Set();
    const hre = new RegExp(p.hint.source, p.hint.flags.includes("i") ? "gi" : "g");
    while ((m = hre.exec(text))) hints.add(m[0].toLowerCase());
    let V = 1;
    lines.forEach(ln => { const hits = ln.match(CONTROL); if (hits) V += hits.length; });
    return {
      L_total: total, L_dma: dma, L_dma_args: dma,
      L_coh: countRe(lines, p.coh), L_sync: countRe(lines, p.sync_inter),
      L_intra_sync: countRe(lines, p.sync_intra), L_pc: countRe(lines, p.pc),
      L_scalar: countRe(lines, p.scalar), L_alloc: countRe(lines, p.alloc),
      L_addr: countRe(lines, p.addr), L_compute: countRe(lines, p.compute),
      L_scale: countRe(lines, p.scale), L_head: firstDma(lines, p.dma_call),
      P_src: countRe(lines, p.pipe_switch), N_alive: Math.max(1, countRe(lines, p.alloc)),
      V_base: V, f_cluster: countRe(lines, p.cluster) > 0 ? 1 : 0,
      N_hint_used: hints.size, api_names: Array.from(new Set(names)),
      N_api: new Set(names).size, N_enum_used: enums.size, files: 1, missing: false
    };
  }

  function zeroMetrics() {
    return {
      L_total: 0, L_dma: 0, L_dma_args: 0, L_coh: 0, L_sync: 0, L_intra_sync: 0,
      L_pc: 0, L_scalar: 0, L_alloc: 0, L_addr: 0, L_compute: 0, L_scale: 0,
      L_head: 0, P_src: 0, N_alive: 0, V_base: 1, f_cluster: 0, N_hint_used: 0,
      api_names: [], N_api: 0, N_enum_used: 0, files: 0, file_list: [], missing: true
    };
  }

  function mergeMetrics(parts) {
    if (!parts.length) return zeroMetrics();
    const out = zeroMetrics();
    out.missing = false;
    const names = [], files = [];
    let cluster = 0, hints = 0, enums = 0;
    parts.forEach(p => {
      Object.keys(out).forEach(k => {
        if (["api_names", "file_list", "missing"].indexOf(k) >= 0) return;
        if (k === "N_alive") out[k] = Math.max(out[k], p[k] || 0);
        else if (k === "f_cluster") cluster = Math.max(cluster, p[k] || 0);
        else if (k === "N_hint_used") hints += p[k] || 0;
        else if (k === "N_enum_used") enums = Math.max(enums, p[k] || 0);
        else if (typeof p[k] === "number") out[k] += p[k];
      });
      (p.api_names || []).forEach(n => names.push(n));
      (p.file_list || []).forEach(n => files.push(n));
    });
    out.api_names = Array.from(new Set(names));
    out.N_api = out.api_names.length;
    out.f_cluster = cluster;
    out.N_hint_used = hints;
    out.N_enum_used = enums;
    out.file_list = files;
    out.V_base = Math.max(1, out.V_base);
    return out;
  }

  function globToRe(g) {
    const s = g.replace(/\\/g, "/").replace(/[.+^${}()|[\]\\]/g, "\\$&")
      .replace(/\*\*/g, "::GLOBSTAR::").replace(/\*/g, "[^/]*").replace(/::GLOBSTAR::/g, ".*");
    return new RegExp(s + "$", "i");
  }
  function matchGlob(path, glob) {
    const p = path.replace(/\\/g, "/");
    try { return globToRe(glob).test(p) || globToRe(glob).test(p.split("/").pop()); }
    catch (e) { return p.toLowerCase().indexOf(glob.replace(/\*/g, "").toLowerCase()) >= 0; }
  }

  function assignFiles(fileEntries, operators, arch) {
    const byOp = {};
    (operators || []).forEach(op => { byOp[String(op.id)] = []; });
    fileEntries.forEach(fe => {
      if (!SRC_EXT.test(fe.name || fe.path || "")) return;
      let hit = null;
      for (const op of (operators || [])) {
        const globs = (op.globs && op.globs[arch]) || [];
        const path = (fe.path || fe.name || "").replace(/\\/g, "/");
        if (globs.some(g => matchGlob(path, g))) { hit = String(op.id); break; }
      }
      if (!hit) {
        const path = ((fe.path || fe.name || "") + " " + (fe.name || "")).toLowerCase();
        for (const op of (operators || [])) {
          const tokens = String(op.op || "").split(/[,/·\s]+/).filter(t => t && t.length >= 3);
          if (tokens.some(t => path.indexOf(t.toLowerCase()) >= 0)) { hit = String(op.id); break; }
        }
      }
      if (hit) byOp[hit].push(fe);
    });
    return byOp;
  }

  function extractFromEntries(fileEntriesByArch, operators) {
    const out = { version: 1, kind: "source_extract", arches: {} };
    ARCHES.forEach(arch => {
      const entries = fileEntriesByArch[arch] || [];
      const byOp = assignFiles(entries, operators, arch);
      const archOut = { arch, root: "(browser)", operators: {} };
      (operators || []).forEach(op => {
        const oid = String(op.id);
        const parts = (byOp[oid] || []).map(fe => {
          const m = analyzeText(fe.text || "", arch);
          m.file_list = [fe.path || fe.name];
          return m;
        });
        archOut.operators[oid] = {
          id: op.id, pattern: op.pattern, op: op.op, weight: op.weight,
          metrics: mergeMetrics(parts)
        };
      });
      out.arches[arch] = archOut;
    });
    return out;
  }

  function present(m) { return m && !m.missing && (m.L_total || 0) > 0; }
  function hwOf(hw, arch) { return (hw && hw[arch]) ? hw[arch] : (hw || {}); }
  function perfOp(perf, arch, oid) { return (((perf || {})[arch] || {})[String(oid)]) || {}; }
  function tier(p, t) { return ((p.tiers || {})[t]) || {}; }
  function opsList(extract, arch) {
    const ops = (((extract || {}).arches || {})[arch] || {}).operators || {};
    return Object.keys(ops).sort((a, b) => Number(a) - Number(b)).map(k => ops[k]);
  }
  function patW(patterns, oid, cat) {
    const p = (patterns || []).find(x => Number(x.id) === Number(oid));
    if (!p) return 1;
    return DIFF_W[(p.difficulty || {})[cat]] || 1;
  }
  function aggOps(extract, arch, perf, patterns, cat, fn) {
    const pairs = [];
    opsList(extract, arch).forEach(op => {
      const s = fn(op.metrics || {}, perfOp(perf, arch, op.id));
      if (s == null) return;
      pairs.push([(op.weight || 1) * patW(patterns, op.id, cat), s]);
    });
    return wavg(pairs);
  }

  function s1a1(hw, p) {
    const N = hw.N || 64;
    const dims = [];
    [["tileSize", 8192, 65536], ["burstLen", 32, 256], ["stride", 1, 64], ["coreConc", 1, N]].forEach(d => {
      const ss = [];
      TIERS.forEach(t => {
        const val = tier(p, t)[d[0]];
        if (val == null) return;
        ss.push(d[0] === "coreConc" ? lnScore(Number(val), 1, N) : lnScore(Number(val), d[1], d[2]));
      });
      if (ss.length) dims.push(avg(ss));
    });
    return avg(dims);
  }
  function s3a1(hw) {
    const bufs = hw.buffers || [];
    if (!bufs.length) return null;
    return avg(bufs.map(b => {
      const s = Number(b.align_B || 0);
      return s <= 0 ? 10 : lnScore(s, 1, 64);
    }));
  }
  function s3a2(m, p) {
    if (p.p0 != null && Number(p.p0) >= 0.95) return 10;
    const Lt = Math.max(1, m.L_total || 1);
    const ss = [];
    TIERS.forEach(t => {
      const T = tier(p, t);
      if (T.Lc == null && T.dV == null) return;
      const Lc = Number(T.Lc || 0), dV = Number(T.dV || 1);
      const Sg = Lt > 1 ? clamp10(10 * (Math.log(Lt) - Math.log(Lc + 1)) / Math.log(Lt)) : 10;
      const Sv = 10 * clamp(2 - Math.exp(dV - 1), 0, 1);
      ss.push(0.6 * Sg + 0.4 * Sv);
    });
    return avg(ss);
  }
  function s3a3(hw) {
    const pref = hw.pipe_ref || 3;
    const keys = ["P_gm2vec", "P_gm2cube", "P_cube2vec", "P_vec2cube"];
    if (!keys.some(k => hw[k] != null)) return null;
    return avg(keys.map(k => 10 * clamp(1 - Number(hw[k] || 0) / pref, 0, 1)));
  }
  function s3a4(m, hw, p) {
    const n = Number(m.N_alive || 1);
    const bufs = hw.buffers || [];
    if (!bufs.length) return null;
    const ss = [];
    TIERS.forEach(t => {
      const T = tier(p, t).tileSize;
      if (T == null) return;
      const bs = [];
      bufs.forEach(b => {
        const cap = Number(b.cap_KiB || 0) * 1024;
        if (cap <= 0) return;
        const rho = n * Number(T) / cap;
        bs.push(rho <= 1 ? 10 : Math.min(10, 10 / rho));
      });
      if (bs.length) ss.push(avg(bs));
    });
    return avg(ss);
  }
  function sLines(num, den, xref) { return present({ L_total: den, missing: false }) ? cost(ratio(num, den), xref) : null; }
  function s3a5(m) { return present(m) ? sLines(m.L_dma, m.L_total, 0.3) : null; }
  function s3b1(m) { return present(m) ? sLines(m.L_coh, m.L_total, 0.2) : null; }
  function s3b2(hw, perf, extract, arch) {
    let nHint = hw.N_hint;
    if (nHint == null) {
      const used = opsList(extract, arch).filter(o => present(o.metrics)).map(o => o.metrics.N_hint_used || 0);
      nHint = used.length ? Math.max.apply(null, used) : null;
    }
    if (nHint == null) return null;
    const S_hint = clamp10(10 - 2 * Number(nHint));
    const gaps = [];
    opsList(extract, arch).forEach(op => {
      const g = perfOp(perf, arch, op.id).hint_gap;
      if (g != null) gaps.push(Number(g));
    });
    let S_gap = null;
    if (gaps.length) {
      const g = avg(gaps);
      S_gap = g < 0.05 ? 10 : g < 0.15 ? 8 : g < 0.3 ? 5 : 3;
    }
    if (S_gap == null) return S_hint;
    return 0.45 * S_hint + 0.55 * S_gap;
  }
  function s3b3(extract, arch, patterns) {
    const pairs = [];
    opsList(extract, arch).forEach(op => {
      if (!present(op.metrics)) return;
      pairs.push([(op.weight || 1) * patW(patterns, op.id, "内存模型"), clamp10(10 - 5 * (op.metrics.f_cluster || 0))]);
    });
    return wavg(pairs);
  }
  function s3b4(hw, perf, extract, arch) {
    const ic = Number(hw.ICache_KiB || 0), dc = Number(hw.DCache_KiB || 0);
    let Sr = null;
    if (ic + dc > 0 && hw.binary_stack_KiB != null) {
      const r = Number(hw.binary_stack_KiB) / (ic + dc);
      Sr = 10 * clamp(1 - Math.max(r - 1, 0) / 0.1, 0, 1);
    }
    const misses = [];
    opsList(extract, arch).forEach(op => {
      const v = perfOp(perf, arch, op.id).icache_miss;
      if (v != null) misses.push(Number(v));
    });
    let Sm = null;
    if (misses.length) {
      const m = avg(misses);
      Sm = m < 0.01 ? 10 : m < 0.05 ? 8 : m < 0.15 ? 5 : 3;
    }
    if (Sr == null && Sm == null) return null;
    if (Sr == null) return Sm;
    if (Sm == null) return Sr;
    return 0.4 * Sr + 0.6 * Sm;
  }
  function s4a1(hw, extract, arch) {
    if (hw.N_api != null && hw.N_viol != null && Number(hw.N_api) > 0)
      return 10 * (1 - Number(hw.N_viol) / Number(hw.N_api));
    let total = 0, viol = 0;
    opsList(extract, arch).forEach(op => {
      if (!present(op.metrics)) return;
      total += op.metrics.N_api || 0;
      viol += Math.min(op.metrics.N_api || 0, (op.metrics.L_sync || 0) + (op.metrics.L_intra_sync || 0));
    });
    return total > 0 ? 10 * (1 - viol / total) : null;
  }
  function s4a2(hw) {
    if (hw.N_hw == null || hw.N_api_dup == null || !hw.N_hw) return null;
    const eta = Number(hw.N_api_dup) / Number(hw.N_hw);
    const Se = 10 * (1 - eta);
    const r = hw.eta_ratio_vs_baseline;
    const Sc = r ? 10 * clamp(1 / r, 0, 1) : Se;
    return 0.65 * Se + 0.35 * Sc;
  }
  function s4a3(hw, extract, arch) {
    const Spi = hw.optional_param_ratio != null ? 10 * clamp(1 - Number(hw.optional_param_ratio) / 0.5, 0, 1) : null;
    let Seps = null;
    if (hw.N_enum_def) {
      const used = opsList(extract, arch).filter(o => present(o.metrics)).map(o => o.metrics.N_enum_used || 0);
      if (used.length) Seps = 10 * clamp((Math.max.apply(null, used) / Number(hw.N_enum_def) - 0.3) / 0.7, 0, 1);
    }
    if (Spi == null && Seps == null) return null;
    if (Spi == null) return Seps;
    if (Seps == null) return Spi;
    return 0.5 * Spi + 0.5 * Seps;
  }
  function s4a4(hw) {
    const keys = ["r_hw", "r_simt", "r_sw", "r_cover"];
    if (!keys.some(k => hw[k] != null)) return null;
    return 0.25 * 10 * Number(hw.r_hw || 0) + 0.25 * 10 * Number(hw.r_simt || 0)
      + 0.15 * 10 * (1 - Number(hw.r_sw || 0)) + 0.35 * 10 * Number(hw.r_cover || 0);
  }
  function s4b1(p) {
    if (p.T_fused == null || !p.T_no_fused) return null;
    return 10 * clamp(1 - Math.abs(Number(p.T_fused) / Number(p.T_no_fused) - 1) / 0.25, 0, 1);
  }
  function s4b2(hw, extract, arch, perf) {
    const parts = [];
    if (hw.r_mma != null) parts.push([0.25, 10 * clamp(Number(hw.r_mma), 0, 1)]);
    if (hw.ulp != null) parts.push([0.15, 10 * clamp(1 - Math.max(Number(hw.ulp) - 2, 0) / 2, 0, 1)]);
    if (hw.c_conv != null) parts.push([0.15, clamp10(10 / Math.pow(2, Number(hw.c_conv)) + 0.5 * Number(hw.round_rtn || 0) + 0.5 * Number(hw.round_sr || 0))]);
    const sp = [], mp = [];
    opsList(extract, arch).forEach(op => {
      const m = op.metrics || {}, w = op.weight || 1;
      if (present(m) && m.L_total) sp.push([w, 10 * clamp(1 - (m.L_scale / m.L_total) / 0.15, 0, 1)]);
      const p = perfOp(perf, arch, op.id);
      if (p.mfu_quant != null && p.mfu_fp16) mp.push([w, 10 * clamp(Number(p.mfu_quant) / Number(p.mfu_fp16), 0, 1)]);
    });
    const Ss = wavg(sp), Sm = wavg(mp);
    if (Ss != null) parts.push([0.15, Ss]);
    if (Sm != null) parts.push([0.20, Sm]);
    if (hw.r_ieee != null) parts.push([0.10, 10 * Number(hw.r_ieee)]);
    const tw = parts.reduce((a, x) => a + x[0], 0);
    return tw ? parts.reduce((a, x) => a + x[0] / tw * x[1], 0) : null;
  }
  function s5a1(m, p) {
    if (!present(m) || p.P_runtime == null || !m.P_src) return null;
    return 10 * clamp(2 - Number(p.P_runtime) / Number(m.P_src), 0, 1);
  }
  function s5b1(m) { return present(m) ? sLines(m.L_sync, m.L_total, 0.3) : null; }
  function s5b2(m) { return present(m) ? sLines(m.L_intra_sync, m.L_total, 0.25) : null; }
  function s5c1(m) { return present(m) ? sLines(m.L_pc, m.L_total, 0.35) : null; }
  function s5d1(m, p) {
    const ss = [];
    TIERS.forEach(t => {
      const T = tier(p, t);
      let Sg = null, Sc = null;
      if (T.T_scalar != null && T.T_maxPipe) Sg = 10 * clamp(1 - (Number(T.T_scalar) / Number(T.T_maxPipe) - 1) / 0.05, 0, 1);
      if (present(m)) Sc = 10 * clamp(1 - (m.L_scalar / m.L_total) / 0.3, 0, 1);
      if (Sg == null && Sc == null) return;
      ss.push(Sg == null ? Sc : Sc == null ? Sg : 0.55 * Sg + 0.45 * Sc);
    });
    return avg(ss);
  }
  function s5e1(m, p) {
    let St = null, Sc = null;
    if (p.T_head != null && p.T_kernel) St = 10 * clamp(1 - Number(p.T_head) / Number(p.T_kernel) / 0.2, 0, 1);
    if (present(m) && m.L_total) Sc = 10 * clamp(1 - m.L_head / m.L_total / 0.25, 0, 1);
    if (St == null && Sc == null) return null;
    if (St == null) return Sc;
    if (Sc == null) return St;
    return 0.55 * St + 0.45 * Sc;
  }
  function s6a1(m) { return present(m) ? sLines(m.L_dma_args, m.L_total, 0.35) : null; }
  function s6a2(m) { return present(m) ? sLines(m.L_addr, m.L_total, 0.3) : null; }
  function s6b1(m) { return present(m) ? sLines(m.L_alloc, m.L_total, 0.25) : null; }
  function rate(hw, n, d, defD) {
    const den = hw[d] != null ? hw[d] : defD;
    if (hw[n] == null || den == null || !Number(den)) return null;
    return 10 * clamp(Number(hw[n]) / Number(den), 0, 1);
  }
  function s7a2(hw) {
    if (hw.p_det == null) return null;
    const noise = 10 * (1 - (Number(hw.p_fn || 0) + Number(hw.p_fp || 0) + Number(hw.p_dup || 0) + Number(hw.p_sdc || 0)) / 4);
    return 0.6 * 10 * Number(hw.p_det) + 0.4 * clamp10(noise);
  }
  function s7a3(hw) {
    if (hw.c_acc == null && hw.c_comp == null) return null;
    if (hw.c_acc == null) return 10 * Number(hw.c_comp);
    if (hw.c_comp == null) return 10 * Number(hw.c_acc);
    return 0.55 * 10 * Number(hw.c_acc) + 0.45 * 10 * Number(hw.c_comp);
  }
  function s7d2(hw) { return hw.e_rel == null ? null : 10 * clamp(1 - Number(hw.e_rel) / 0.1, 0, 1); }
  function s7d3(hw) { return hw.r_loss == null ? null : 10 * clamp((Number(hw.r_loss) - 0.95) / 0.05, 0, 1); }
  function s7d6(hw) {
    const St = hw.delta_t != null ? 10 * clamp(1 - Number(hw.delta_t) / 0.05, 0, 1) : null;
    const Sp = hw.delta_b != null ? 10 * clamp(1 - Number(hw.delta_b) / 0.02, 0, 1) : null;
    if (St == null && Sp == null) return null;
    if (St == null) return Sp;
    if (Sp == null) return St;
    return 0.55 * St + 0.45 * Sp;
  }
  function s8(p) {
    if (p.P_agent == null || !p.P_theory) return null;
    return 10 * clamp(Number(p.P_agent) / Number(p.P_theory), 0, 1);
  }
  function s9(p) {
    const vers = p.versions || [];
    let n = p.N_versions;
    if (n == null) n = vers.length || null;
    if (n == null) return null;
    const Sn = 10 * clamp(1 - (Number(n) - 1) / 5, 0, 1);
    const g = [], r = [];
    for (let i = 1; i < vers.length; i++) {
      const prev = vers[i - 1], cur = vers[i];
      const L0 = Number(prev.lines || 0);
      const dL = cur.delta_lines != null ? Number(cur.delta_lines) : Math.abs(Number(cur.lines || 0) - L0);
      if (L0 > 0) g.push(dL / L0);
      if (prev.time && cur.time) r.push(Number(cur.time) / Number(prev.time));
    }
    const Sg = g.length ? 10 * clamp(1 - avg(g) / 0.3, 0, 1) : null;
    const Sr = r.length ? 10 * clamp((1 - avg(r)) / 0.3, 0, 1) : null;
    const parts = [[0.25, Sn], [0.35, Sg], [0.40, Sr]].filter(x => x[1] != null);
    const tw = parts.reduce((a, x) => a + x[0], 0);
    return tw ? parts.reduce((a, x) => a + x[0] / tw * x[1], 0) : null;
  }

  function scoreArch(arch, extract, hwAll, perf, patterns) {
    const hw = hwOf(hwAll, arch);
    const o = {};
    o["1.a.1"] = aggOps(extract, arch, perf, patterns, "系统规格", (m, p) => s1a1(hw, p));
    o["3.a.1"] = s3a1(hw);
    o["3.a.2"] = aggOps(extract, arch, perf, patterns, "内存模型", s3a2);
    o["3.a.3"] = s3a3(hw);
    o["3.a.4"] = aggOps(extract, arch, perf, patterns, "内存模型", (m, p) => s3a4(m, hw, p));
    o["3.a.5"] = aggOps(extract, arch, perf, patterns, "内存模型", (m) => s3a5(m));
    o["3.b.1"] = aggOps(extract, arch, perf, patterns, "内存模型", (m) => s3b1(m));
    o["3.b.2"] = s3b2(hw, perf, extract, arch);
    o["3.b.3"] = s3b3(extract, arch, patterns);
    o["3.b.4"] = s3b4(hw, perf, extract, arch);
    o["4.a.1"] = s4a1(hw, extract, arch);
    o["4.a.2"] = s4a2(hw);
    o["4.a.3"] = s4a3(hw, extract, arch);
    o["4.a.4"] = s4a4(hw);
    o["4.b.1"] = aggOps(extract, arch, perf, patterns, "计算模型", (m, p) => s4b1(p));
    o["4.b.2"] = s4b2(hw, extract, arch, perf);
    o["5.a.1"] = aggOps(extract, arch, perf, patterns, "控制模型", s5a1);
    o["5.b.1"] = aggOps(extract, arch, perf, patterns, "控制模型", (m) => s5b1(m));
    o["5.b.2"] = aggOps(extract, arch, perf, patterns, "控制模型", (m) => s5b2(m));
    o["5.c.1"] = aggOps(extract, arch, perf, patterns, "控制模型", (m) => s5c1(m));
    o["5.d.1"] = aggOps(extract, arch, perf, patterns, "控制模型", s5d1);
    o["5.e.1"] = aggOps(extract, arch, perf, patterns, "控制模型", s5e1);
    o["6.a.1"] = aggOps(extract, arch, perf, patterns, "访存模型", (m) => s6a1(m));
    o["6.a.2"] = aggOps(extract, arch, perf, patterns, "访存模型", (m) => s6a2(m));
    o["6.b.1"] = aggOps(extract, arch, perf, patterns, "访存模型", (m) => s6b1(m));
    o["7.a.1"] = rate(hw, "exc_detected", "exc_defined");
    o["7.a.2"] = s7a2(hw);
    o["7.a.3"] = s7a3(hw);
    o["7.a.4"] = hw.r_root != null ? 10 * Number(hw.r_root) : rate(hw, "root_ok", "root_total");
    o["7.b.1"] = hw.bp_supported != null ? 10 * clamp(Number(hw.bp_supported) / 12, 0, 1) : null;
    o["7.b.2"] = hw.r_loc != null ? 10 * Number(hw.r_loc) : rate(hw, "bp_loc_ok", "bp_loc_total");
    o["7.b.3"] = hw.r_pc != null ? 10 * Number(hw.r_pc) : rate(hw, "pc_ok", "pc_total");
    o["7.b.4"] = hw.r_halt != null ? 10 * Number(hw.r_halt) : rate(hw, "halt_ok", "halt_total");
    o["7.c.1"] = hw.r_obscov != null ? 10 * Number(hw.r_obscov) : rate(hw, "obs_ok", "obs_total");
    o["7.d.1"] = hw.r_evtcov != null ? 10 * Number(hw.r_evtcov) : rate(hw, "evt_ok", "evt_total");
    o["7.d.2"] = s7d2(hw);
    o["7.d.3"] = s7d3(hw);
    o["7.d.4"] = hw.bn_ok != null ? 10 * clamp(Number(hw.bn_ok) / 9, 0, 1) : null;
    o["7.d.5"] = hw.r_expl != null ? 10 * Number(hw.r_expl) : null;
    o["7.d.6"] = s7d6(hw);
    o["7.d.7"] = hw.r_tasklink != null ? 10 * Number(hw.r_tasklink) : null;
    o["8.a.1"] = aggOps(extract, arch, perf, patterns, "Agent友好", (m, p) => s8(p));
    o["9.a.1"] = aggOps(extract, arch, perf, patterns, "开发效率", (m, p) => s9(p));
    return o;
  }

  function scoreAll(extract, hw, perf, patterns) {
    const scores = {}, by_arch = {};
    ARCHES.forEach(arch => {
      const a = scoreArch(arch, extract, hw, perf, patterns);
      by_arch[arch] = a;
      Object.keys(a).forEach(code => {
        if (!scores[code]) scores[code] = {};
        scores[code][arch] = a[code] == null ? null : Math.round(a[code] * 100) / 100;
      });
    });
    return { kind: "computed_scores", scores, by_arch };
  }

  function defaultHw() {
    return {
      N: 0, Bpeak_GBs: 0, cacheline_B: 64, tile_min: 8192, tile_max: 65536,
      burst_min: 32, burst_max: 256,
      buffers: [
        { name: "UB", align_B: 0, cap_KiB: 0 },
        { name: "L1", align_B: 0, cap_KiB: 0 },
        { name: "L0A", align_B: 0, cap_KiB: 0 },
        { name: "L0B", align_B: 0, cap_KiB: 0 },
        { name: "L0C", align_B: 0, cap_KiB: 0 }
      ],
      ICache_KiB: 0, DCache_KiB: 0, pipe_ref: 3,
      P_gm2vec: 0, P_gm2cube: 0, P_cube2vec: 0, P_vec2cube: 0,
      N_hint: 0, binary_stack_KiB: 0,
      N_api: 0, N_viol: 0, N_hw: 0, N_api_dup: 0,
      optional_param_ratio: 0, N_enum_def: 0,
      r_hw: 0, r_simt: 0, r_sw: 0, r_cover: 0,
      r_mma: 0, ulp: 2, c_conv: 0, round_rtn: 0, round_sr: 0, r_ieee: 0,
      exc_detected: 0, exc_defined: 0, p_det: 0, p_fn: 0, p_fp: 0, p_dup: 0, p_sdc: 0,
      c_acc: 0, c_comp: 0, r_root: 0, bp_supported: 0,
      r_loc: 0, r_pc: 0, r_halt: 0, r_obscov: 0, r_evtcov: 0, e_rel: 0, r_loss: 1,
      bn_ok: 0, r_expl: 0, delta_t: 0, delta_b: 0, r_tasklink: 0
    };
  }

  function defaultPerfOp() {
    const tiers = {};
    TIERS.forEach(t => {
      tiers[t] = { tileSize: "", burstLen: "", stride: "", coreConc: "", Lc: "", dV: "", T_scalar: "", T_maxPipe: "" };
    });
    return {
      p0: "", hint_gap: "", icache_miss: "", T_fused: "", T_no_fused: "",
      P_runtime: "", T_head: "", T_kernel: "", P_agent: "", P_theory: "",
      mfu_quant: "", mfu_fp16: "", N_versions: "",
      tiers
    };
  }

  global.MeasureLib = {
    ARCHES, TIERS, SOURCE_KIND, KIND_LABEL, analyzeText, extractFromEntries,
    scoreAll, defaultHw, defaultPerfOp, present, assignFiles, SRC_EXT
  };
})(typeof window !== "undefined" ? window : globalThis);
