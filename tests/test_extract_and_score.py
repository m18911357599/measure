#!/usr/bin/env python3
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_source import analyze_text, extract_all, match_globs  # noqa: E402
from score import (  # noqa: E402
    clamp,
    ln_score,
    score_1_a_1_op,
    score_3_a_1,
    score_3_a_5_op,
    score_5_b_1_op,
    score_all,
    score_ratio_lines,
)


class LnScoreTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(ln_score(8192, 8192, 65536), 10.0)
        self.assertEqual(ln_score(65536, 8192, 65536), 0.0)
        mid = ln_score(32768, 8192, 65536)
        self.assertTrue(0 < mid < 10)

    def test_cost(self):
        self.assertEqual(score_ratio_lines(0, 100, 0.3), 10.0)
        self.assertEqual(score_ratio_lines(30, 100, 0.3), 0.0)
        self.assertAlmostEqual(score_ratio_lines(15, 100, 0.3), 5.0)


class ExtractTests(unittest.TestCase):
    def test_ascendc_fixture(self):
        text = (ROOT / "tests/fixtures/ascendc_gelu.cpp").read_text()
        m = analyze_text(text, "AscendC")
        self.assertGreater(m["L_total"], 5)
        self.assertGreater(m["L_dma"], 0)
        self.assertGreater(m["L_alloc"], 0)
        self.assertGreater(m["P_src"], 0)
        self.assertGreaterEqual(m["V_base"], 1)
        self.assertLess(m["L_head"], m["L_total"])

    def test_superscalar_fixture(self):
        text = (ROOT / "tests/fixtures/superscalar_matmul.hpp").read_text()
        m = analyze_text(text, "SuperScalar")
        self.assertGreater(m["L_dma"], 0)
        self.assertGreater(m["L_compute"], 0)
        self.assertGreater(m["N_alive"], 0)

    def test_simt_fixture(self):
        text = (ROOT / "tests/fixtures/simt_gelu.cu").read_text()
        m = analyze_text(text, "SIMT")
        self.assertGreater(m["L_dma"], 0)
        self.assertGreater(m["L_sync"] + m["L_intra_sync"], 0)
        self.assertGreater(m["L_alloc"], 0)

    def test_extract_all_from_fixtures(self):
        fx = ROOT / "tests/fixtures"
        data = extract_all({
            "AscendC": str(fx),
            "SuperScalar": str(fx),
            "SIMT": str(fx),
        })
        a = data["arches"]["AscendC"]["operators"]["1"]["metrics"]
        self.assertFalse(a["missing"])
        self.assertGreater(a["L_total"], 0)
        s = data["arches"]["SuperScalar"]["operators"]["4"]["metrics"]
        self.assertFalse(s["missing"])
        t = data["arches"]["SIMT"]["operators"]["1"]["metrics"]
        self.assertFalse(t["missing"])
        # Cross-arch filenames must not leak into the wrong ISA extract.
        self.assertTrue(data["arches"]["SuperScalar"]["operators"]["1"]["metrics"]["missing"])
        self.assertTrue(data["arches"]["AscendC"]["operators"]["4"]["metrics"]["missing"])
        self.assertTrue(data["arches"]["SIMT"]["operators"]["4"]["metrics"]["missing"])


class ScoreIntegrationTests(unittest.TestCase):
    def test_hw_align(self):
        hw = {"buffers": [{"name": "UB", "align_B": 1}, {"name": "L1", "align_B": 64}]}
        s = score_3_a_1(hw)
        self.assertAlmostEqual(s, 5.0)

    def test_source_ratio_and_all(self):
        fx = ROOT / "tests/fixtures"
        extract = extract_all({
            "AscendC": str(fx),
            "SuperScalar": str(fx),
            "SIMT": str(fx),
        })
        m = extract["arches"]["AscendC"]["operators"]["1"]["metrics"]
        s = score_3_a_5_op(m)
        self.assertIsNotNone(s)
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 10)

        hw = {
            "AscendC": {
                "N": 32,
                "buffers": [{"name": "UB", "align_B": 32, "cap_KiB": 192}],
                "pipe_ref": 3,
                "P_gm2vec": 1,
                "P_gm2cube": 1,
                "P_cube2vec": 0,
                "P_vec2cube": 0,
                "N_hint": 1,
                "exc_detected": 8,
                "exc_defined": 10,
                "bp_supported": 6,
                "r_hw": 0.6,
                "r_simt": 0.2,
                "r_sw": 0.1,
                "r_cover": 0.5,
            },
            "SuperScalar": {"N": 8, "buffers": [{"name": "UB", "align_B": 1, "cap_KiB": 256}]},
            "SIMT": {"N": 108, "buffers": [{"name": "smem", "align_B": 16, "cap_KiB": 164}]},
        }
        perf = {
            "AscendC": {
                "1": {
                    "p0": 0.97,
                    "tiers": {
                        "60%": {"tileSize": 8192, "burstLen": 32, "stride": 1, "coreConc": 1},
                        "90%": {"tileSize": 16384, "burstLen": 64, "stride": 8, "coreConc": 4},
                        "99%": {"tileSize": 32768, "burstLen": 128, "stride": 32, "coreConc": 16},
                    },
                    "T_fused": 1.0,
                    "T_no_fused": 1.0,
                    "P_agent": 0.8,
                    "P_theory": 1.0,
                }
            }
        }
        computed = score_all(extract, hw, perf, None)
        self.assertIn("1.a.1", computed["scores"])
        self.assertIsNotNone(computed["scores"]["3.a.5"]["AscendC"])
        self.assertGreaterEqual(computed["scores"]["3.a.5"]["AscendC"], 0)
        self.assertEqual(computed["scores"]["7.a.1"]["AscendC"], 8.0)
        self.assertEqual(computed["scores"]["7.b.1"]["AscendC"], 5.0)
        # p0>=95% → 3.a.2 is 10 for op 1; other ops missing so weighted = 10
        self.assertEqual(computed["scores"]["3.a.2"]["AscendC"], 10.0)

    def test_empty_hw_does_not_score(self):
        extract = extract_all({"AscendC": str(ROOT / "tests/fixtures")})
        hw = {
            "AscendC": {
                "N": "",
                "buffers": [{"name": "UB", "align_B": "", "cap_KiB": ""}],
                "pipe_ref": 3,
            }
        }
        computed = score_all(extract, hw, {}, None)
        self.assertIsNone(computed["scores"]["3.a.1"]["AscendC"])
        self.assertIsNone(computed["scores"]["4.a.1"]["AscendC"])
        self.assertIsNone(computed["scores"]["7.a.1"]["AscendC"])
        self.assertIsNone(computed["scores"]["7.b.1"]["AscendC"])
        self.assertIsNotNone(computed["scores"]["3.a.5"]["AscendC"])

    def test_bandwidth_formula(self):
        hw = {"N": 64}
        perf = {
            "tiers": {
                "60%": {"tileSize": 8192, "burstLen": 32, "stride": 1, "coreConc": 1},
                "90%": {"tileSize": 8192, "burstLen": 32, "stride": 1, "coreConc": 1},
                "99%": {"tileSize": 8192, "burstLen": 32, "stride": 1, "coreConc": 1},
            }
        }
        self.assertAlmostEqual(score_1_a_1_op(hw, perf), 10.0)


class LocalOpsTests(unittest.TestCase):
    def test_expand_windows_drive_on_posix(self):
        import os
        from local_ops import expand_local_path
        if os.name == "nt":
            p = expand_local_path(r"D:\cursor\measure")
            self.assertIsNotNone(p)
            return
        self.assertIsNone(expand_local_path(r"D:\cursor\measure"))
        self.assertIsNone(expand_local_path("D:/cursor/measure"))
        self.assertTrue(expand_local_path(str(ROOT / "tests" / "fixtures")).is_dir())

    def test_discover_repo_falls_back_to_fixtures(self):
        from local_ops import discover_roots, find_cache_root, local_status
        cache = find_cache_root(ROOT)
        self.assertTrue((cache / "measure.md").is_file())
        roots = discover_roots(cache, ROOT)
        for arch in ("AscendC", "SuperScalar", "SIMT"):
            p = Path(roots[arch])
            self.assertTrue(p.is_dir(), arch)
        ascend = Path(roots["AscendC"]).resolve()
        if (ROOT / "ops-nn").is_dir() and (ROOT / "ops-math").is_dir():
            self.assertEqual(ascend, ROOT.resolve())
        else:
            self.assertEqual(Path(roots["AscendC"]).name, "fixtures")
        super_p = Path(roots["SuperScalar"])
        if (ROOT / "SuperNpuBench").is_dir():
            self.assertNotEqual(super_p.name, "fixtures")
        else:
            self.assertEqual(super_p.name, "fixtures")
        simt_p = Path(roots["SIMT"])
        gemm = ROOT / "gemm-cuda"
        if gemm.is_dir() and any(gemm.rglob("*.cu")):
            self.assertEqual(simt_p.name, "gemm-cuda")
        else:
            self.assertEqual(simt_p.name, "fixtures")
        st = local_status(ROOT)
        self.assertTrue(st["exists"]["AscendC"])

    def test_discover_ops_layout(self):
        import shutil
        import tempfile
        from local_ops import discover_roots
        td = Path(tempfile.mkdtemp())
        try:
            (td / "measure.md").write_text("# m\n", encoding="utf-8")
            (td / "ops-nn" / "gelu" / "op_kernel").mkdir(parents=True)
            (td / "ops-math").mkdir()
            sol_parent = td / "SuperNpuBench" / "benchmark" / "one-level-arch" / "kernels"
            sol = sol_parent / "solution"
            sol.mkdir(parents=True)
            (td / "cuda").mkdir()
            (td / "gemm-cuda").mkdir()
            roots = discover_roots(td, ROOT)
            self.assertEqual(Path(roots["AscendC"]).resolve(), td.resolve())
            self.assertEqual(Path(roots["SuperScalar"]).resolve(), sol_parent.resolve())
            self.assertEqual(Path(roots["SIMT"]).resolve(), (td / "gemm-cuda").resolve())
        finally:
            shutil.rmtree(td)

    def test_extract_local_cli(self):
        import subprocess
        out = ROOT / "tests" / "_tmp_extract_local.json"
        try:
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "measure.py"), "extract", "--local", "--out", str(out)],
                cwd=str(ROOT), capture_output=True, text=True, check=False,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(data["arches"]["AscendC"]["operators"]["1"]["metrics"]["missing"])
            self.assertFalse(data["arches"]["SuperScalar"]["operators"]["4"]["metrics"]["missing"])
            simt = data["arches"]["SIMT"]["operators"]
            found = [oid for oid, o in simt.items() if not o["metrics"].get("missing")]
            self.assertTrue(found, "SIMT should have at least one operator")
            root_name = Path(data["arches"]["SIMT"]["root"]).name
            if root_name in ("fixtures", "tests"):
                self.assertIn("1", found)
            elif root_name == "gemm-cuda":
                self.assertIn("4", found)
        finally:
            if out.exists():
                out.unlink()

    def test_cann_globs_cover_underscore_names(self):
        if not (ROOT / "ops-math" / "math" / "arg_max_v2").is_dir():
            self.skipTest("vendored CANN kernels not present")
        from extract_source import load_operators
        operators = load_operators()
        data = extract_all({"AscendC": str(ROOT)}, ROOT / "scripts" / "operators.json")
        missing = [
            o["id"]
            for o in data["arches"]["AscendC"]["operators"].values()
            if o["metrics"].get("missing")
        ]
        self.assertEqual(missing, [], f"AscendC missing patterns {missing}")
        by_id = {str(o["id"]): o for o in operators["operators"]}
        for oid, needle in (("5", "arg_max"), ("11", "affine_grid"), ("21", "drop_out")):
            files = " ".join(data["arches"]["AscendC"]["operators"][oid]["metrics"].get("file_list") or [])
            self.assertIn(needle, files.replace("\\", "/"), f"#{oid} {by_id[oid]['pattern']}")

    def test_merge_keeps_placeholders_and_records_extract(self):
        from score import merge_into_measure_data, score_all
        fx = ROOT / "tests" / "fixtures"
        extract = extract_all({"AscendC": str(fx)})
        computed = score_all(extract, {}, {}, None)
        measure = {
            "meta": {},
            "scores": {
                "1.a.1": {"AscendC": {"score": 5, "source": "profiling"}},
                "7.a.1": {"AscendC": {"score": 5, "source": "hardware"}},
            },
        }
        merged = merge_into_measure_data(measure, computed, extract)
        self.assertEqual(merged["scores"]["1.a.1"]["AscendC"]["score"], 5)
        self.assertEqual(merged["scores"]["1.a.1"]["AscendC"]["source"], "profiling")
        self.assertEqual(merged["scores"]["7.a.1"]["AscendC"]["source"], "hardware")
        self.assertEqual(merged["scores"]["3.a.5"]["AscendC"]["source"], "computed")
        self.assertIsNotNone(merged["scores"]["3.a.5"]["AscendC"]["score"])
        self.assertIn("extract", merged)
        self.assertGreaterEqual(merged["meta"]["local_measure"]["coverage"]["AscendC"]["found"], 1)

    def test_super_extract_from_vendored_kernels(self):
        kernels = (
            ROOT / "SuperNpuBench" / "benchmark" / "one-level-arch" / "kernels"
        )
        if not (kernels / "solution" / "matmul_test").is_dir():
            self.skipTest("vendored SuperNpuBench kernels not present")
        data = extract_all({"SuperScalar": str(kernels)}, ROOT / "scripts" / "operators.json")
        ops = data["arches"]["SuperScalar"]["operators"]
        present = [oid for oid, o in ops.items() if not o["metrics"].get("missing")]
        for oid in ("1", "4", "6", "8", "13", "18", "22", "25", "26"):
            self.assertIn(oid, present, f"SuperScalar #{oid} should have source")
        self.assertTrue(ops["23"]["metrics"].get("missing"), "sparse.mm must not steal QSMLA")
        self.assertTrue(ops["20"]["metrics"].get("missing"), "dynamic-batch must not steal mx quant")
        files4 = " ".join(ops["4"]["metrics"].get("file_list") or [])
        self.assertIn("matmul", files4.replace("\\", "/"))


class SimtGemmCudaTests(unittest.TestCase):
    def test_auth_url_embeds_token_and_redacts(self):
        from download_cann_ops import _redact_cmd, simt_auth_url
        url = "https://gitcode.com/mlidongfeng/gemm-cuda.git"
        auth = simt_auth_url(url, "secret-token", "oauth2")
        self.assertIn("secret-token", auth)
        self.assertTrue(auth.startswith("https://oauth2:secret-token@gitcode.com/"))
        red = " ".join(_redact_cmd(["git", "clone", auth, "dest"]))
        self.assertNotIn("secret-token", red)
        self.assertIn("oauth2:***@", red)
        self.assertEqual(simt_auth_url(url, ""), url)

    def test_copy_and_map_gemm_files(self):
        import shutil
        import tempfile
        from download_cann_ops import copy_simt_tree, map_simt_patterns
        from extract_source import load_operators
        td = Path(tempfile.mkdtemp())
        try:
            src = td / "src"
            (src / "src").mkdir(parents=True)
            (src / "README.md").write_text("# gemm-cuda\n", encoding="utf-8")
            (src / "src" / "sgemm.cu").write_text(
                "__global__ void sgemm() { __syncthreads(); }\n", encoding="utf-8"
            )
            (src / "tests" / "bench.cu").parent.mkdir(parents=True)
            (src / "tests" / "bench.cu").write_text("// test only\n", encoding="utf-8")
            dst = td / "gemm-cuda"
            n = copy_simt_tree(src, dst, include_optional=False)
            self.assertGreaterEqual(n, 2)
            self.assertTrue((dst / "src" / "sgemm.cu").is_file())
            self.assertFalse((dst / "tests" / "bench.cu").is_file())
            stats = map_simt_patterns(dst, load_operators())
            self.assertGreater(stats["4"]["files"], 0)
            self.assertTrue(any("sgemm" in p for p in stats["4"]["present"]))
        finally:
            shutil.rmtree(td)

    def test_gemm_cuda_does_not_leak_into_ascendc(self):
        import shutil
        import tempfile
        td = Path(tempfile.mkdtemp())
        try:
            gelu = td / "ops-nn" / "activation" / "gelu" / "op_kernel"
            gelu.mkdir(parents=True)
            gelu_src = (ROOT / "tests" / "fixtures" / "ascendc_gelu.cpp").read_text(encoding="utf-8")
            (gelu / "gelu.cpp").write_text(gelu_src, encoding="utf-8")
            (td / "gemm-cuda").mkdir()
            (td / "gemm-cuda" / "gelu.cu").write_text(
                "__global__ void gelu() {}\n", encoding="utf-8"
            )
            (td / "gemm-cuda" / "sgemm.cu").write_text(
                "__global__ void sgemm() { __syncthreads(); }\n", encoding="utf-8"
            )
            data = extract_all(
                {"AscendC": str(td), "SIMT": str(td / "gemm-cuda")},
                ROOT / "scripts" / "operators.json",
            )
            afiles = " ".join(
                data["arches"]["AscendC"]["operators"]["1"]["metrics"].get("file_list") or []
            )
            self.assertNotIn(".cu", afiles)
            self.assertFalse(data["arches"]["SIMT"]["operators"]["4"]["metrics"].get("missing"))
            sfiles = " ".join(
                data["arches"]["SIMT"]["operators"]["4"]["metrics"].get("file_list") or []
            )
            self.assertIn("sgemm.cu", sfiles.replace("\\", "/"))
        finally:
            shutil.rmtree(td)

    def test_simt_extract_from_vendored_gemm_cuda(self):
        gemm = ROOT / "gemm-cuda"
        if not gemm.is_dir() or not any(gemm.rglob("*.cu")):
            self.skipTest("vendored gemm-cuda kernels not present")
        data = extract_all({"SIMT": str(gemm)}, ROOT / "scripts" / "operators.json")
        ops = data["arches"]["SIMT"]["operators"]
        present = [oid for oid, o in ops.items() if not o["metrics"].get("missing")]
        self.assertTrue(present, "vendored gemm-cuda should match at least one pattern")
        self.assertIn("4", present)


if __name__ == "__main__":
    unittest.main()
