"""Markdown rendering. Labels follow your output contract: what was counted vs. what is a reading of it."""
from __future__ import annotations

from .families import SPLITS

MEANING = {
    "replay": "same items it was shown",
    "iid": "new items, same kinds",
    "extrap": "new items, bigger numbers / longer inputs",
    "novel": "kinds of task never shown",
    "paraphrase": "iid items reworded",
}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def render_run(meta, m):
    L = [f"# Generalization run: {meta['system']}", "",
         f"seed {meta['seed']}, {meta['n']} items per family per split, learned from {meta['learned']} training items"
         + ("" if meta["learned"] else " (system has no learning hook, so replay is just another test)"), "",
         "## Accuracy by split (I counted)", "", "| split | meaning | correct / n | accuracy | 95% interval |", "|---|---|---|---|---|"]
    for s in SPLITS:
        d = m["splits"][s]
        L.append(f"| {s} | {MEANING[s]} | {d['correct']}/{d['n']} | {_pct(d['acc'])} | {_pct(d['ci'][0])} to {_pct(d['ci'][1])} |")
    L += ["", f"Errors raised by the system (counted as wrong): {m['errors']}", "", "## Gaps (computed from the table)"]
    reading = {
        "replay_minus_iid": "recall of seen items minus accuracy on new ones; large means recall without generalization",
        "iid_minus_extrap": "accuracy drop at larger scale",
        "iid_minus_novel": "accuracy drop on kinds of task it was not shown",
        "iid_minus_paraphrase": "accuracy drop when only the wording changes",
    }
    for k, v in m["gaps"].items():
        L.append(f"- {k}: {100 * v:+.0f} points ({reading[k]})")
    L.append(f"- items right as written but wrong when reworded: {m['paraphrase_flips']}")
    L += ["", "## By family and split (I counted)", "",
          "| family | " + " | ".join(f"{s} acc (95% CI)" for s in SPLITS) + " |",
          "|---|" + "---|" * len(SPLITS)]
    for fam, d in sorted(m["families"].items()):
        cells = []
        for s in SPLITS:
            entry = d.get(s)
            if entry is None:
                cells.append("n/a")
            elif isinstance(entry, dict):
                acc = entry.get("acc")
                ci = entry.get("ci", (None, None))
                if acc is None:
                    cells.append("n/a")
                else:
                    cells.append(f"{_pct(acc)} ({_pct(ci[0])}–{_pct(ci[1])})")
            else:
                # legacy float (from a saved run produced by an older version)
                cells.append(_pct(entry))
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def render_compare(meta_a, meta_b, c):
    lo, hi = c["ci"]
    L = [f"# A vs B: {meta_a['system']}  vs  {meta_b['system']}", "",
         f"{c['n_items']} paired items (same seed and n for both runs).", "",
         "## Result (I counted; the verdict line is a reading of the numbers)", "",
         f"- Mean accuracy difference, B minus A: {100 * c['mean_diff']:+.1f} points "
         f"(95% bootstrap interval {100 * lo:+.1f} to {100 * hi:+.1f})",
         f"- Right only with A: {c['a_only_correct']}; right only with B: {c['b_only_correct']}; "
         f"right with both: {c['both_correct']}; wrong with both: {c['neither_correct']}",
         f"- Identical outputs (ignoring surrounding whitespace): {100 * c['identical_output_rate']:.1f}% of items; "
         f"exactly byte-identical: {100 * c['exact_identical_rate']:.1f}%",
         "", f"**Verdict:** {c['verdict']}", "", "## Difference by split (B minus A)"]
    for s, d in c["per_split_diff"].items():
        L.append(f"- {s}: {100 * d:+.0f} points")
    L += ["", "Reading note: a difference here is a difference in these task families only. It says nothing about "
              "tasks outside them."]
    return "\n".join(L) + "\n"
