# generalize — evaluation harness

Standalone, model-independent evaluation package that measures whether a
system (a bare model checkpoint, or a model wrapped in additional machinery
such as retrieval or an agent layer) generalises beyond training.

## What it measures

Five splits, each answering a different question:

| split | question |
|---|---|
| replay | Does the system recall items it was shown? |
| iid | Does it solve new items from the same task families? |
| extrap | Does it hold up with larger numbers / longer inputs? |
| novel | Does it transfer to task families it was never shown? |
| paraphrase | Is it wording-sensitive (same numbers, different phrasing)? |

A paired comparison (`generalize compare`) tests whether a wrapper (e.g. a
retrieval layer, a prompt-engineering step, or an Ava-like agent) changes the
answers at all and, if so, whether the change is measurably good or bad.

## Quick start

```bash
# Calibration smoke-test — no model needed
python -m generalize selftest

# Run a local Ollama model
python -m generalize run --system ollama:llama3.2 --n 10 --out my_run

# Compare two runs
python -m generalize compare my_run.json other_run.json --out diff.md
```

## Adapters

| spec prefix | example | notes |
|---|---|---|
| `ollama:<model>` | `ollama:llama3.2` | calls local Ollama server |
| `callable:<module>:<func>` | `callable:solver_adapter:answer` | imports a Python function |
| `cmd:<command>` | `cmd:./my_model.sh` | calls a subprocess |

`ava_adapter_template.py` — research template for an Ava-wrapped model; unwired
until the real Ava entrypoint is available.

`expansformer_adapter_template.py` — interface contract for a live Expansformer
checkpoint adapter; **not wired to any checkpoint**. See the integration
checklist inside the file.

## Evidence status

- 25/25 contract tests pass (run: `python -m pytest tests/ -v`)
- Tested on calibration systems whose correct outputs are known (oracle,
  memorizer, constant-0, wording-bound solver)
- **Not tested against a real Expansformer checkpoint** — that requires wiring
  `expansformer_adapter_template.py` to a live inference path

## Integration gaps

1. `expansformer_adapter_template.py` — needs wiring to an actual checkpoint
   inference path before use in a real evaluation run
2. `ava_adapter_template.py` — kept as a research template; Ava integration is
   not present in this repository (status: planned/intended)
3. No task families covering Expansformer's specific training domains yet —
   existing families (arithmetic, sequences, word problems, logic, string ops)
   test general-purpose reasoning only
4. `task_manifest_hash` is recorded in run JSON but not yet checked by CI;
   add a step that compares hashes before accepting a comparison as paired

## Isolation contract

- Pure Python 3.10+; zero non-stdlib dependencies
- No imports from `expansformer_model/`, training code, or checkpoints
- No writes outside the output path passed via `--out`
- Safe to run in a read-only repo checkout (tests use `tempfile.TemporaryDirectory`)
