"""Run a system over the splits, compute metrics, and compare two systems item by item."""
from __future__ import annotations

import hashlib
import math
import random
import re

from .families import SPLITS


def extract(raw, kind):
    t = (raw or "").strip()
    if not t:
        return ""
    if kind == "int":
        nums = re.findall(r"-?\d+", t.replace(",", ""))
        return nums[-1] if nums else ""
    last = [ln for ln in t.splitlines() if ln.strip()][-1].strip()
    last = re.sub(r"^(answer|result)\s*[:=]\s*", "", last, flags=re.I)
    return last.strip("`'\". ").lower()


def run_system(system, tasks, learn=True):
    """Return (rows_by_split, learned_count). Errors are recorded per item and count as wrong.

    Leakage guard: raises RuntimeError if any evaluation split contains a prompt that is
    identical to a training prompt. This detects adapter bugs where eval items are accidentally
    mixed into the training batch.
    """
    train_prompts = {t.prompt for t in tasks.get("train", [])}
    # replay is *by design* the training items re-presented; skip it in the leakage check.
    # Only the held-out splits must not contain training prompts.
    held_out = [s for s in SPLITS if s != "replay"]
    for split in held_out:
        for t in tasks.get(split, []):
            if t.prompt in train_prompts:
                raise RuntimeError(
                    f"Evaluation leakage detected: item {t.id!r} in split {split!r} has the same "
                    f"prompt as a training item. The held-out evaluation set must not overlap with training."
                )

    learned = 0
    if learn and system.supports_learning:
        for t in tasks["train"]:
            system.learn(t.prompt, t.answer)
            learned += 1
    rows = {}
    for split in SPLITS:
        rows[split] = []
        for t in tasks[split]:
            err = None
            try:
                raw = system.answer(t.prompt)
            except Exception as e:  # noqa: BLE001 - recorded on the item
                raw, err = "", repr(e)
            pred = extract(raw, t.kind)
            rows[split].append({
                "id": t.id, "family": t.family, "split": split, "raw": raw, "pred": pred,
                "answer": t.answer.lower(), "correct": pred == t.answer.lower().strip(), "error": err,
            })
    return rows, learned


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def metrics(rows):
    out = {"splits": {}, "families": {}, "errors": 0}
    for split, rs in rows.items():
        k = sum(r["correct"] for r in rs)
        lo, hi = wilson(k, len(rs))
        out["splits"][split] = {"n": len(rs), "correct": k, "acc": k / len(rs) if rs else None, "ci": (lo, hi)}
        out["errors"] += sum(1 for r in rs if r["error"])
        for fam in {r["family"] for r in rs}:
            fr = [r for r in rs if r["family"] == fam]
            fk = sum(r["correct"] for r in fr)
            flo, fhi = wilson(fk, len(fr))
            out["families"].setdefault(fam, {})[split] = {
                "n": len(fr), "correct": fk,
                "acc": fk / len(fr) if fr else None,
                "ci": (flo, fhi),
            }
    acc = lambda s: out["splits"][s]["acc"]  # noqa: E731
    g = {}
    if None not in (acc("replay"), acc("iid")):
        g["replay_minus_iid"] = acc("replay") - acc("iid")       # large => recall without generalization
    if None not in (acc("iid"), acc("extrap")):
        g["iid_minus_extrap"] = acc("iid") - acc("extrap")       # large => breaks at larger scale
    if None not in (acc("iid"), acc("novel")):
        g["iid_minus_novel"] = acc("iid") - acc("novel")         # large => does not transfer to new tasks
    if None not in (acc("iid"), acc("paraphrase")):
        g["iid_minus_paraphrase"] = acc("iid") - acc("paraphrase")  # large => wording-sensitive
    iid_ok = {r["id"] for r in rows["iid"] if r["correct"]}
    flips = [r for r in rows["paraphrase"] if r["id"][:-2] in iid_ok and not r["correct"]]
    out["paraphrase_flips"] = len(flips)
    out["gaps"] = g
    return out


def _flat(rows):
    return {r["id"] + "|" + r["split"]: r for rs in rows.values() for r in rs}


def compare(rows_a, rows_b, resamples=2000, seed=0, require_identical_sets=True):
    """Paired comparison of B against A on identical items.

    By default (require_identical_sets=True) raises ValueError when A and B were not evaluated
    on exactly the same items. A partial overlap silently hides missing coverage in one system,
    so the default is strict. Pass require_identical_sets=False only when you understand which
    items are being compared and why the sets differ.
    """
    fa, fb = _flat(rows_a), _flat(rows_b)
    only_a = sorted(set(fa) - set(fb))
    only_b = sorted(set(fb) - set(fa))
    if only_a or only_b:
        if require_identical_sets:
            raise ValueError(
                f"evaluation sets are not identical: {len(only_a)} item(s) present only in A, "
                f"{len(only_b)} item(s) present only in B. "
                f"Use the same --seed and --n for both runs. "
                f"Pass require_identical_sets=False to compare on the intersection only."
            )
    keys = sorted(set(fa) & set(fb))
    if not keys:
        raise ValueError("no shared items: both runs must use the same --seed and --n")
    diffs = [int(fb[k]["correct"]) - int(fa[k]["correct"]) for k in keys]
    h = lambda s: hashlib.sha256(s.encode()).digest()  # noqa: E731
    exact = sum(1 for k in keys if h(fa[k]["raw"]) == h(fb[k]["raw"]))
    ident = sum(1 for k in keys if h(fa[k]["raw"].strip()) == h(fb[k]["raw"].strip()))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples))
    lo, hi = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    mean = sum(diffs) / len(diffs)
    per_split = {}
    for split in SPLITS:
        ks = [k for k in keys if k.endswith("|" + split)]
        if ks:
            per_split[split] = sum(int(fb[k]["correct"]) - int(fa[k]["correct"]) for k in ks) / len(ks)
    res = {
        "n_items": len(keys),
        "missing_from_b": len(only_a),
        "missing_from_a": len(only_b),
        "mean_diff": mean, "ci": (lo, hi),
        "a_only_correct": sum(1 for k in keys if fa[k]["correct"] and not fb[k]["correct"]),
        "b_only_correct": sum(1 for k in keys if fb[k]["correct"] and not fa[k]["correct"]),
        "both_correct": sum(1 for k in keys if fa[k]["correct"] and fb[k]["correct"]),
        "neither_correct": sum(1 for k in keys if not fa[k]["correct"] and not fb[k]["correct"]),
        "identical_output_rate": ident / len(keys),      # equal after trimming surrounding whitespace
        "exact_identical_rate": exact / len(keys),       # equal byte for byte
        "per_split_diff": per_split,
    }
    if exact == len(keys):
        res["verdict"] = ("B's outputs are byte-identical to A's on every item. Whatever B adds is not "
                          "reaching the answer.")
    elif ident == len(keys):
        res["verdict"] = ("B's outputs are identical to A's on every item apart from leading or trailing "
                          "whitespace. Whatever B adds is not reaching the answer.")
    elif lo > 0:
        res["verdict"] = "B is measurably better than A on these items (the interval excludes 0)."
    elif hi < 0:
        res["verdict"] = "B is measurably worse than A on these items (the interval excludes 0)."
    else:
        res["verdict"] = "No measurable difference at this sample size (the interval includes 0)."
        if res["identical_output_rate"] >= 0.95:
            res["verdict"] += (f" Outputs match on {100 * res['identical_output_rate']:.0f}% of items, "
                               "so B is changing very little.")
    return res
