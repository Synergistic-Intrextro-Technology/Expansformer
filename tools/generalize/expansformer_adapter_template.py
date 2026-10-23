"""Expansformer-native adapter interface — UNWIRED TEMPLATE.

This file defines the interface that a live Expansformer checkpoint adapter
must satisfy to be used with `generalize run --system callable:...`.

Status: TEMPLATE ONLY — not wired to any running checkpoint, training job,
or inference server. Do not use this in a real evaluation run until an actual
entrypoint is implemented and the path below updated.

Integration checklist (fill in when wiring):
  - [ ] Identify the checkpoint path or inference server URL
  - [ ] Confirm tokenizer and context-length are compatible with eval prompts
  - [ ] Implement answer() to call that entrypoint and return raw string output
  - [ ] Decide whether learn() is meaningful (fine-tune step, KV-cache write, etc.)
  - [ ] Set supports_learning = True only if learn() actually modifies state
  - [ ] Run `python -m generalize selftest` and confirm the harness can
        distinguish oracle vs. memorizor vs. constant baselines through this adapter
  - [ ] Record checkpoint path, model version, and config hash in the run meta
        by passing --system callable:generalize.expansformer_adapter:ExpansformerSystem
        after implementing the class below

Expansformer-Ava boundary note:
  This adapter belongs to Expansformer (the model layer). Ava is a planned
  product/agent layer that may call Expansformer via an interface like this,
  but Ava integration is not present in this repository. Do not import Ava
  modules here until a concrete integration design is implemented and the
  project instructions are updated to reflect it.
"""
from __future__ import annotations


class ExpansformerSystem:
    """Adapter wrapping a live Expansformer checkpoint for use with the
    generalize evaluation harness.

    To wire this up:
      1. Replace the NotImplementedError stubs with real calls to your
         inference path (local model load, gRPC call, REST call, etc.).
      2. Set `name` to something that uniquely identifies the checkpoint
         (e.g. "expansformer-v0.4-step12000").
      3. Update `supports_learning` once learn() has a real implementation.
    """

    name: str = "expansformer-UNWIRED"
    supports_learning: bool = False  # set True once learn() is implemented

    # ------------------------------------------------------------------ #
    # Required interface                                                   #
    # ------------------------------------------------------------------ #

    def answer(self, prompt: str) -> str:
        """Call the Expansformer checkpoint and return its raw string output.

        The generalize harness calls extract() on the returned string to
        parse the final answer token, so returning the full model output
        (including any chain-of-thought) is safe and recommended.

        Wire this to: model forward pass, inference server, or subprocess.
        """
        raise NotImplementedError(
            "Wire this to the Expansformer inference entrypoint. "
            "See integration checklist at the top of this file."
        )

    def learn(self, prompt: str, correct_answer: str) -> None:
        """Optionally expose a learning hook for the replay/iid splits.

        Leave this as a no-op (and keep supports_learning = False) unless
        Expansformer has a genuine online-learning or few-shot-tuning path
        that you want to test against those splits.

        If implemented, this should NOT modify the checkpoint on disk; it
        should only affect in-memory state for the duration of one run()
        call, so that runs are reproducible.
        """
        # Example stub — replace with real implementation if/when available:
        # self._model.few_shot_update(prompt, correct_answer)
        pass


# ------------------------------------------------------------------ #
# Callable entry-point expected by make_system()                     #
# ------------------------------------------------------------------ #

def make_expansformer_system() -> ExpansformerSystem:
    """Entry point for --system callable:generalize.expansformer_adapter_template:make_expansformer_system

    Replace with real initialisation once the adapter is wired:
      - load checkpoint
      - initialise tokenizer
      - warm up inference path
      - return configured ExpansformerSystem instance
    """
    return ExpansformerSystem()
