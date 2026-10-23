"""Procedural task families and the splits built from them.

Splits (each answers a different question):
  train       items a learning system may study (never scored)
  replay      the exact train items again          -> recall / memorization
  iid         new items, same families and scale   -> in-distribution generalization
  extrap      new items, larger numbers/longer     -> extrapolation
  novel       families never in train              -> transfer to new kinds of task
  paraphrase  iid items reworded                   -> sensitivity to surface wording
Everything is deterministic from (seed, n).
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

SUFFIX = " Reply with only the answer."


@dataclass
class Task:
    id: str
    family: str
    split: str
    prompt: str
    answer: str
    kind: str  # "int" or "text"
    alt_prompt: str = ""


# ---- train families -------------------------------------------------------------------------

def gen_arith(rng, large):
    op = rng.choice("+-*")
    if not large:
        a, b = rng.randint(0, 20), rng.randint(0, 20)
    elif op == "*":
        a, b = rng.randint(20, 99), rng.randint(20, 99)
    else:
        a, b = rng.randint(100, 999), rng.randint(100, 999)
    ans = {"+": a + b, "-": a - b, "*": a * b}[op]
    alt = {"+": f"What is {a} plus {b}?", "-": f"Subtract {b} from {a}.", "*": f"What is {a} times {b}?"}[op]
    return f"Compute {a} {op} {b}.", alt, str(ans), "int"


def gen_seq(rng, large):
    if rng.random() < 0.3:
        r = rng.choice([2, 3])
        s = rng.randint(10, 99) if large else rng.randint(1, 9)
        n = 5 if large else 4
        terms, nxt = [s * r**i for i in range(n)], s * r**n
    else:
        step = rng.randint(10, 99) if large else rng.randint(1, 9)
        s = rng.randint(100, 999) if large else rng.randint(0, 20)
        n = 6 if large else 4
        terms, nxt = [s + step * i for i in range(n)], s + step * n
    seq = ", ".join(map(str, terms))
    return f"What number comes next: {seq}, ?", f"Continue the sequence {seq} with one more number.", str(nxt), "int"


_STRING_OPS = {
    "reverse": (lambda s: s[::-1], "Reverse the string '{s}'.", "Write the text '{s}' backwards."),
    "upper": (lambda s: s.upper(), "Uppercase the string '{s}'.", "Convert '{s}' to capital letters."),
    "sort": (lambda s: "".join(sorted(s)), "Sort the letters of '{s}' alphabetically.",
             "Rearrange the characters of '{s}' into alphabetical order."),
    "rotate": (lambda s: s[1:] + s[:1], "Rotate '{s}' left by one character.",
               "Move the first character of '{s}' to the end."),
}


def gen_string(rng, large):
    s = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(10, 14) if large else rng.randint(4, 6)))
    fn, p, alt = _STRING_OPS[rng.choice(list(_STRING_OPS))]
    return p.format(s=s), alt.format(s=s), fn(s), "text"


_LIST_OPS = {
    "sum": (sum, "Add up the numbers {xs}.", "What is the total of {xs}?"),
    "max": (max, "What is the largest number in {xs}?", "Find the maximum of {xs}."),
    "min": (min, "What is the smallest number in {xs}?", "Find the minimum of {xs}."),
    "even": (lambda xs: sum(1 for x in xs if x % 2 == 0), "How many numbers in {xs} are even?",
             "Count the even values in {xs}."),
    "second": (lambda xs: sorted(xs)[-2], "What is the second largest number in {xs}?",
               "Find the runner-up maximum of {xs}."),
}


def gen_listop(rng, large):
    xs = rng.sample(range(1000 if large else 21), rng.randint(10, 14) if large else rng.randint(4, 6))
    fn, p, alt = _LIST_OPS[rng.choice(list(_LIST_OPS))]
    return p.format(xs=xs), alt.format(xs=xs), str(fn(xs)), "int"


# ---- novel families (never in train) --------------------------------------------------------

def gen_grid(rng, large):
    g = [[rng.randint(0, 3) for _ in range(3)] for _ in range(3)]
    op = rng.choice(["fliph", "transpose", "swap12"])
    if op == "fliph":
        out, p, alt = [r[::-1] for r in g], "flip horizontally", "Mirror each row left to right."
    elif op == "transpose":
        out, p, alt = [list(c) for c in zip(*g)], "transpose", "Swap rows and columns."
    else:
        out, p, alt = [[{1: 2, 2: 1}.get(v, v) for v in r] for r in g], "swap the digits 1 and 2", "Replace every 1 with 2 and every 2 with 1."
    enc = lambda grid: "/".join("".join(map(str, r)) for r in grid)  # noqa: E731
    return (
        f"Grid (rows separated by /): {enc(g)}. Apply: {p}. Give the resulting grid in the same format.",
        f"Rows: {' | '.join(''.join(map(str, r)) for r in g)}. {alt} Answer in the same / format.",
        enc(out),
        "text",
    )


def gen_induct(rng, large):
    kind = rng.choice(["add", "mul", "lin"])
    k = rng.randint(2, 9)
    f = {"add": lambda x: x + k, "mul": lambda x: k * x, "lin": lambda x: 2 * x + k}[kind]
    xs = rng.sample(range(1, 13), 3)
    q = rng.randint(13, 20)
    shots = ", ".join(f"f({x})={f(x)}" for x in xs)
    pairs = ", ".join(f"{x}->{f(x)}" for x in xs)
    return f"{shots}. What is f({q})?", f"Given the pairs {pairs}, find the output for {q}.", str(f(q)), "int"


def gen_compose(rng, large):
    n, a, b = rng.randint(1, 20), rng.randint(1, 9), rng.randint(2, 5)
    if rng.random() < 0.5:
        return (f"Start with {n}. Add {a}, then multiply the result by {b}.",
                f"Take {n}, add {a} to it, then multiply the sum by {b}.", str((n + a) * b), "int")
    return (f"Start with {n}. Multiply by {b}, then add {a}.",
            f"Take {n}, multiply it by {b}, then add {a} to the product.", str(n * b + a), "int")


TRAIN_FAMILIES = {"arith": gen_arith, "seq": gen_seq, "string": gen_string, "listop": gen_listop}
NOVEL_FAMILIES = {"grid": gen_grid, "induct": gen_induct, "compose": gen_compose}
SPLITS = ("replay", "iid", "extrap", "novel", "paraphrase")


def _draw(gen, rng, large, n, avoid, tag, family, split):
    out, tries = [], 0
    while len(out) < n and tries < 2000:
        tries += 1
        p, alt, ans, kind = gen(rng, large)
        full = p + SUFFIX
        if full in avoid:
            continue
        avoid.add(full)
        out.append(Task(f"{family}-{tag}-{len(out)}", family, split, full, ans, kind, alt + SUFFIX))
    return out


def build_tasks(seed, n):
    """Return {split: [Task]} plus 'train'. n items per (family, split)."""
    rng = random.Random(seed)
    tasks = {k: [] for k in ("train",) + SPLITS}
    seen = set()
    for fam, gen in TRAIN_FAMILIES.items():
        train = _draw(gen, rng, False, n, seen, "train", fam, "train")
        tasks["train"] += train
        tasks["replay"] += [Task(t.id, t.family, "replay", t.prompt, t.answer, t.kind, t.alt_prompt) for t in train]
        iid = _draw(gen, rng, False, n, seen, "iid", fam, "iid")
        tasks["iid"] += iid
        tasks["extrap"] += _draw(gen, rng, True, n, seen, "extrap", fam, "extrap")
        tasks["paraphrase"] += [Task(t.id + "~p", fam, "paraphrase", t.alt_prompt, t.answer, t.kind) for t in iid]
    for fam, gen in NOVEL_FAMILIES.items():
        tasks["novel"] += _draw(gen, rng, False, n, seen, "novel", fam, "novel")
    return tasks
