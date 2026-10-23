"""An independent, parse-based solver for the PRIMARY prompt wordings.

It exists for two reasons: (1) to check the harness's ground truth against a second implementation
that shares no code with the generators, and (2) to act as a real system that is bound to one
wording, so the paraphrase split has something genuine to catch. Paraphrased prompts return '?'.
"""
import re

SUFFIX = " Reply with only the answer."


def solve(prompt):
    p = prompt.replace(SUFFIX, "").strip()

    m = re.fullmatch(r"Compute (-?\d+) ([+\-*]) (-?\d+)\.", p)
    if m:
        a, op, b = int(m[1]), m[2], int(m[3])
        return str({"+": a + b, "-": a - b, "*": a * b}[op])

    m = re.fullmatch(r"What number comes next: (.*), \?", p)
    if m:
        xs = [int(x) for x in m[1].split(", ")]
        d = xs[1] - xs[0]
        if all(xs[i + 1] - xs[i] == d for i in range(len(xs) - 1)):
            return str(xs[-1] + d)
        r = xs[1] // xs[0]
        if all(xs[i + 1] == xs[i] * r for i in range(len(xs) - 1)):
            return str(xs[-1] * r)
        return "?"

    m = re.search(r"'([a-z]+)'", p)
    if m and re.match(r"(Reverse|Uppercase|Sort|Rotate)", p):
        s = m[1]
        if p.startswith("Reverse"):
            return s[::-1]
        if p.startswith("Uppercase"):
            return s.upper()
        if p.startswith("Sort"):
            return "".join(sorted(s))
        return s[1:] + s[:1]

    m = re.search(r"\[(.*?)\]", p)
    if m:
        xs = [int(x) for x in m[1].split(", ")]
        if p.startswith("Add up"):
            return str(sum(xs))
        if "second largest" in p:
            return str(sorted(xs)[-2])
        if "largest" in p:
            return str(max(xs))
        if "smallest" in p:
            return str(min(xs))
        if "even" in p:
            return str(sum(1 for x in xs if x % 2 == 0))

    m = re.match(r"Grid \(rows separated by /\): ([0-3/]+)\. Apply: (.*)\. Give", p)
    if m:
        rows = [list(r) for r in m[1].split("/")]
        op = m[2]
        if op == "flip horizontally":
            rows = [r[::-1] for r in rows]
        elif op == "transpose":
            rows = [list(c) for c in zip(*rows)]
        elif op == "swap the digits 1 and 2":
            rows = [[{"1": "2", "2": "1"}.get(v, v) for v in r] for r in rows]
        else:
            return "?"
        return "/".join("".join(r) for r in rows)

    m = re.fullmatch(r"((?:f\(\d+\)=-?\d+(?:, )?)+)\. What is f\((\d+)\)\?", p)
    if m:
        shots = [(int(x), int(y)) for x, y in re.findall(r"f\((\d+)\)=(-?\d+)", m[1])]
        q = int(m[2])
        for k in range(2, 10):
            for fn in (lambda x, k=k: x + k, lambda x, k=k: k * x, lambda x, k=k: 2 * x + k):
                if all(fn(x) == y for x, y in shots):
                    return str(fn(q))
        return "?"

    m = re.fullmatch(r"Start with (\d+)\. Add (\d+), then multiply the result by (\d+)\.", p)
    if m:
        n, a, b = map(int, m.groups())
        return str((n + a) * b)
    m = re.fullmatch(r"Start with (\d+)\. Multiply by (\d+), then add (\d+)\.", p)
    if m:
        n, b, a = map(int, m.groups())
        return str(n * b + a)
    return "?"
