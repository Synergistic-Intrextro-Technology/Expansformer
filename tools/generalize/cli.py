"""python -m generalize run|compare|selftest"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys

from . import __version__
from .evaluate import compare, metrics, run_system
from .families import build_tasks
from .report import render_compare, render_run
from .systems import (ConstantSystem, MemorizerSystem, TableSystem, WrappedSystem, make_system)


def _task_manifest_hash(tasks):
    """SHA-256 (first 16 hex chars) of the canonically serialised evaluation items.

    Recorded in every run so two result files can be compared only when they
    used exactly the same task manifest — same prompts, answers, ids, splits.
    """
    items = []
    for split in sorted(tasks):
        for t in tasks[split]:
            items.append({"id": t.id, "split": split, "prompt": t.prompt, "answer": t.answer})
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:16]


def cmd_run(a):
    tasks = build_tasks(a.seed, a.n)
    system = make_system(a.system)
    rows, learned = run_system(system, tasks, learn=not a.no_learn)
    m = metrics(rows)
    meta = {
        "system": a.system,
        "seed": a.seed,
        "n": a.n,
        "learned": learned,
        "date": datetime.date.today().isoformat(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "generalize_version": __version__,
        "task_manifest_hash": _task_manifest_hash(tasks),
    }
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write(render_run(meta, m))
    print(f"wrote {a.out}.json and {a.out}.md")
    for s, d in m["splits"].items():
        print(f"  {s:11s} {d['correct']}/{d['n']}")
    return 0


def cmd_compare(a):
    with open(a.a, encoding="utf-8") as fa, open(a.b, encoding="utf-8") as fb:
        ra, rb = json.load(fa), json.load(fb)
    if (ra["meta"]["seed"], ra["meta"]["n"]) != (rb["meta"]["seed"], rb["meta"]["n"]):
        print("refusing to compare: the two runs used different --seed/--n", file=sys.stderr)
        return 2
    # Warn when task manifests differ (different prompts/answers despite matching seed/n).
    ha = ra["meta"].get("task_manifest_hash")
    hb = rb["meta"].get("task_manifest_hash")
    if ha and hb and ha != hb:
        print(
            f"warning: task_manifest_hash differs ({ha} vs {hb}). "
            "The two runs may have used different families or family versions.",
            file=sys.stderr,
        )
    try:
        c = compare(ra["rows"], rb["rows"])
    except ValueError as e:
        print(f"refusing to compare: {e}", file=sys.stderr)
        return 2
    text = render_compare(ra["meta"], rb["meta"], c)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


def cmd_selftest(a):
    """Run systems whose behavior is known and show the harness separates them."""
    tasks = build_tasks(a.seed, a.n)
    table = {t.prompt: t.answer for split in tasks.values() for t in split}
    oracle = TableSystem(table, "oracle")
    oracle_no_para = TableSystem({t.prompt: t.answer for k, v in tasks.items() if k != "paraphrase" for t in v}, "surface-bound")
    for s in (oracle, oracle_no_para, MemorizerSystem(), ConstantSystem("0")):
        rows, _ = run_system(s, tasks)
        m = metrics(rows)
        print(f"{s.name:14s} " + "  ".join(f"{k}={100 * d['acc']:.0f}%" for k, d in m["splits"].items()))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="generalize")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one system over all splits")
    r.add_argument("--system", required=True, help="ollama:<model> | callable:<module>:<func> | cmd:<command>")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--n", type=int, default=10, help="items per family per split")
    r.add_argument("--no-learn", action="store_true", help="skip the learning phase even if the system has one")
    r.add_argument("--out", default="run")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare", help="paired comparison of two saved runs")
    c.add_argument("a"), c.add_argument("b")
    c.add_argument("--out", default="compare.md")
    c.set_defaults(fn=cmd_compare)
    s = sub.add_parser("selftest", help="calibration on systems with known behavior")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--n", type=int, default=10)
    s.set_defaults(fn=cmd_selftest)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
