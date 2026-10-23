"""Contract tests for the generalization harness. Run from /home/claude:
    python3 -m unittest discover -s tests -v
Each class states the difference it exists to prove.
"""
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from generalize import cli, evaluate, systems
from generalize.families import NOVEL_FAMILIES, SPLITS, TRAIN_FAMILIES, build_tasks
from generalize.systems import ConstantSystem, MemorizerSystem, System, TableSystem, WrappedSystem
from solver import solve

SEED, N = 7, 12


def all_tasks(tasks):
    return [t for split in tasks.values() for t in split]


def table_for(tasks, include_paraphrase=True):
    return {t.prompt: t.answer for k, v in tasks.items() if include_paraphrase or k != "paraphrase" for t in v}


class GroundTruth(unittest.TestCase):
    """Difference: an independent parse-based solver (no shared code) agrees with every generated
    answer on the primary wording. If the generators had a wrong answer, every result would be corrupt."""

    def test_independent_solver_matches_all_primary_answers(self):
        tasks = build_tasks(SEED, 40)
        bad = [(t.id, t.prompt, t.answer, solve(t.prompt)) for k in ("train", "iid", "extrap", "novel")
               for t in tasks[k] if solve(t.prompt) != t.answer]
        self.assertEqual(bad, [], f"{len(bad)} disagreements, e.g. {bad[:2]}")

    def test_paraphrases_keep_the_same_numbers(self):
        """Weak check (paraphrase meaning was also reviewed by hand): numeric families keep every number."""
        for t in build_tasks(SEED, 40)["iid"]:
            if t.family in ("string", "grid"):
                continue
            nums = lambda s: sorted(re.findall(r"\d+", s.replace(" Reply with only the answer.", "")))  # noqa: E731
            self.assertEqual(nums(t.prompt), nums(t.alt_prompt), t.prompt)


class SplitHygiene(unittest.TestCase):
    """Difference: the splits really are separate, so a gap between them means something."""

    def setUp(self):
        self.t = build_tasks(SEED, N)

    def test_deterministic_by_seed(self):
        a = [(x.id, x.prompt, x.answer) for x in all_tasks(build_tasks(SEED, N))]
        b = [(x.id, x.prompt, x.answer) for x in all_tasks(build_tasks(SEED, N))]
        c = [(x.id, x.prompt, x.answer) for x in all_tasks(build_tasks(SEED + 1, N))]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_no_leakage_into_held_out_splits(self):
        train = {t.prompt for t in self.t["train"]}
        for split in ("iid", "extrap", "novel"):
            self.assertFalse(train & {t.prompt for t in self.t[split]}, split)
        self.assertEqual({t.prompt for t in self.t["replay"]}, train)
        self.assertFalse({t.prompt for t in self.t["paraphrase"]} & train)

    def test_no_duplicate_prompts_within_a_split(self):
        for split in ("iid", "extrap", "novel", "train"):
            ps = [t.prompt for t in self.t[split]]
            self.assertEqual(len(ps), len(set(ps)), split)

    def test_family_sets_are_disjoint_and_counts_exact(self):
        self.assertFalse(set(TRAIN_FAMILIES) & set(NOVEL_FAMILIES))
        self.assertFalse({t.family for t in self.t["train"]} & {t.family for t in self.t["novel"]})
        self.assertEqual(len(self.t["iid"]), N * len(TRAIN_FAMILIES))
        self.assertEqual(len(self.t["novel"]), N * len(NOVEL_FAMILIES))
        self.assertEqual(len(self.t["paraphrase"]), len(self.t["iid"]))

    def test_extrapolation_items_are_actually_larger(self):
        nums = lambda ts: max(int(x) for t in ts for x in re.findall(r"\d+", t.prompt))  # noqa: E731
        self.assertGreater(nums(self.t["extrap"]), 3 * nums(self.t["iid"]))


class Calibration(unittest.TestCase):
    """Difference: systems with known behavior land where they should, so the metrics can be trusted
    to separate recall, generalization and wording-sensitivity."""

    def setUp(self):
        self.tasks = build_tasks(SEED, N)

    def metrics_for(self, system):
        rows, learned = evaluate.run_system(system, self.tasks)
        return evaluate.metrics(rows), learned

    def test_oracle_scores_everywhere(self):
        m, _ = self.metrics_for(TableSystem(table_for(self.tasks)))
        for s in SPLITS:
            self.assertEqual(m["splits"][s]["acc"], 1.0, s)
        self.assertEqual(m["paraphrase_flips"], 0)

    def test_memorizer_shows_recall_without_generalization(self):
        m, learned = self.metrics_for(MemorizerSystem())
        self.assertEqual(learned, len(self.tasks["train"]))
        self.assertEqual(m["splits"]["replay"]["acc"], 1.0)
        self.assertLessEqual(m["splits"]["iid"]["acc"], 0.3)
        self.assertGreaterEqual(m["gaps"]["replay_minus_iid"], 0.6)

    def test_wording_bound_system_is_caught_by_paraphrase_split(self):
        m, _ = self.metrics_for(TableSystem(table_for(self.tasks, include_paraphrase=False)))
        self.assertEqual(m["splits"]["iid"]["acc"], 1.0)
        self.assertEqual(m["splits"]["paraphrase"]["acc"], 0.0)
        self.assertEqual(m["paraphrase_flips"], len(self.tasks["iid"]))

    def test_real_parse_based_solver_is_wording_bound_but_generalizes_in_scale(self):
        class Solver(System):
            name = "solver"

            def answer(self, p):
                return solve(p)

        m, _ = self.metrics_for(Solver())
        for s in ("replay", "iid", "extrap", "novel"):
            self.assertEqual(m["splits"][s]["acc"], 1.0, s)
        # Not exactly 0: this solver matches on the keyword "even", so one reworded listop prompt still
        # solves. The split measured that correctly. What matters is the large drop and the flips.
        self.assertLess(m["splits"]["paraphrase"]["acc"], 0.15)
        self.assertGreaterEqual(m["paraphrase_flips"], 0.85 * len(self.tasks["iid"]))

    def test_constant_system_is_low_everywhere(self):
        m, _ = self.metrics_for(ConstantSystem("0"))
        for s in SPLITS:
            self.assertLess(m["splits"][s]["acc"], 0.35, s)

    def test_errors_are_recorded_and_count_as_wrong(self):
        class Boom(System):
            def answer(self, p):
                raise RuntimeError("down")

        rows, _ = evaluate.run_system(Boom(), self.tasks)
        m = evaluate.metrics(rows)
        self.assertEqual(m["splits"]["iid"]["acc"], 0.0)
        self.assertEqual(m["errors"], sum(len(v) for v in rows.values()))
        self.assertIn("down", rows["iid"][0]["error"])


class WrapperToggle(unittest.TestCase):
    """Difference (your toggle test): a wrapper that changes answers shows a measurable effect;
    a wrapper whose effect never reaches the answer shows byte-identical output and a flagged verdict."""

    def setUp(self):
        self.tasks = build_tasks(SEED, N)
        self.table = table_for(self.tasks)
        self.base = ConstantSystem("0")

    def rows(self, system):
        return evaluate.run_system(system, self.tasks)[0]

    def test_helpful_wrapper_is_detected(self):
        helpful = WrappedSystem(self.base, lambda p, out: self.table[p], "helpful")
        c = evaluate.compare(self.rows(self.base), self.rows(helpful))
        self.assertGreater(c["ci"][0], 0)
        self.assertIn("measurably better", c["verdict"])
        self.assertLess(c["identical_output_rate"], 0.5)
        self.assertEqual(c["a_only_correct"], 0)

    def test_severed_wrapper_is_flagged_as_not_reaching_the_answer(self):
        severed = WrappedSystem(self.base, lambda p, out: out, "severed")
        c = evaluate.compare(self.rows(self.base), self.rows(severed))
        self.assertEqual(c["identical_output_rate"], 1.0)
        self.assertEqual(c["mean_diff"], 0.0)
        self.assertIn("byte-identical", c["verdict"])

    def test_harmful_wrapper_is_detected(self):
        good = TableSystem(self.table, "good")
        harmful = WrappedSystem(good, lambda p, out: "0", "harmful")
        c = evaluate.compare(self.rows(good), self.rows(harmful))
        self.assertLess(c["ci"][1], 0)
        self.assertIn("measurably worse", c["verdict"])

    def test_noise_level_difference_is_not_called_significant(self):
        """A wrapper that fixes one item out of many must not be reported as a real improvement."""
        target = self.tasks["iid"][0].prompt
        slight = WrappedSystem(self.base, lambda p, out: self.table[p] if p == target else out, "slight")
        c = evaluate.compare(self.rows(self.base), self.rows(slight))
        self.assertLessEqual(c["ci"][0], 0)
        self.assertIn("No measurable difference", c["verdict"])

    def test_compare_requires_shared_items(self):
        other = evaluate.run_system(self.base, build_tasks(SEED + 99, N))[0]
        with self.assertRaises(ValueError):
            evaluate.compare({"iid": [{"id": "zzz", "split": "iid", "raw": "", "correct": False}]}, {"iid": other["iid"][:0]})


class Extraction(unittest.TestCase):
    """Difference: reply formatting noise does not change whether an answer counts."""

    def test_int(self):
        self.assertEqual(evaluate.extract("The answer is 1,234.", "int"), "1234")
        self.assertEqual(evaluate.extract("17 + 5 = 22", "int"), "22")
        self.assertEqual(evaluate.extract("-7", "int"), "-7")
        self.assertEqual(evaluate.extract("no idea", "int"), "")
        self.assertEqual(evaluate.extract("", "int"), "")

    def test_text(self):
        self.assertEqual(evaluate.extract("Answer: 'EDCBA'.", "text"), "edcba")
        self.assertEqual(evaluate.extract("thinking...\n`cdeab`", "text"), "cdeab")
        self.assertEqual(evaluate.extract("012/301", "text"), "012/301")

    def test_wilson_interval(self):
        lo, hi = evaluate.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.237, places=2)
        self.assertAlmostEqual(hi, 0.763, places=2)
        self.assertEqual(evaluate.wilson(0, 0), (0.0, 0.0))


class FakeOllama(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        FakeOllama.seen = {"path": self.path, "body": json.loads(self.rfile.read(n))}
        out = json.dumps({"response": "42"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class Adapters(unittest.TestCase):
    """Difference: the three real adapters carry a prompt in and an answer out through the live path."""

    def test_ollama_request_shape(self):
        srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = systems.OllamaSystem("gemma3:1b", host=f"127.0.0.1:{srv.server_port}").answer("hi")
        finally:
            srv.shutdown()
        self.assertEqual(out, "42")
        self.assertEqual(FakeOllama.seen["path"], "/api/generate")
        b = FakeOllama.seen["body"]
        self.assertEqual((b["model"], b["prompt"], b["stream"]), ("gemma3:1b", "hi", False))
        self.assertEqual(b["options"]["temperature"], 0)

    def test_unknown_spec_fails_loudly(self):
        with self.assertRaises(ValueError):
            systems.make_system("nonsense:x")

    def test_callable_without_learn_reports_no_learning(self):
        s = systems.make_system("callable:const_adapter:answer")
        self.assertFalse(s.supports_learning)
        self.assertEqual(s.answer("x"), "0")


class EndToEndCLI(unittest.TestCase):
    """Difference: the real command-line path (run -> files -> compare) works with callable and
    command adapters, and reports a real system as better than a constant one."""

    def test_run_and_compare_through_cli(self):
        cwd = os.getcwd()
        os.chdir(HERE)  # callable adapters are imported from the working directory
        try:
            with tempfile.TemporaryDirectory() as tmp:
                good, bad, cmd = (os.path.join(tmp, x) for x in ("good", "bad", "cmd"))
                self.assertEqual(cli.main(["run", "--system", "callable:solver_adapter:answer", "--n", "3", "--out", good]), 0)
                self.assertEqual(cli.main(["run", "--system", "callable:const_adapter:answer", "--n", "3", "--out", bad]), 0)
                self.assertEqual(cli.main(["run", "--system", f"cmd:{sys.executable} {os.path.join(HERE, 'solver_cmd.py')}",
                                           "--n", "3", "--out", cmd]), 0)
                out = os.path.join(tmp, "cmp.md")
                self.assertEqual(cli.main(["compare", bad + ".json", good + ".json", "--out", out]), 0)
                text = open(out, encoding="utf-8").read()
                self.assertIn("measurably better", text)
                self.assertIn("paired items", text)
                # same solver behind both adapters; the command adds a trailing newline, nothing else
                self.assertEqual(cli.main(["compare", good + ".json", cmd + ".json", "--out", out]), 0)
                text = open(out, encoding="utf-8").read()
                self.assertIn("apart from leading or trailing whitespace", text)
                self.assertIn("exactly byte-identical: 0.0%", text)
                report = open(good + ".md", encoding="utf-8").read()
                self.assertIn("Accuracy by split", report)
                self.assertIn("| iid | new items, same kinds | 12/12 | 100% |", report)
                # mismatched seeds are refused
                other = os.path.join(tmp, "other")
                cli.main(["run", "--system", "callable:const_adapter:answer", "--n", "3", "--seed", "99", "--out", other])
                self.assertEqual(cli.main(["compare", good + ".json", other + ".json", "--out", out]), 2)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
