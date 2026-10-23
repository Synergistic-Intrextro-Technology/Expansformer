"""Systems under test. A system has answer(prompt) -> str and, optionally, learn(prompt, answer).

Real adapters:  ollama:<model>   callable:<module>:<func>   cmd:<shell command>
Calibration systems (behavior known in advance): used to prove the harness can tell them apart.
"""
from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
import urllib.request


class System:
    name = "system"
    supports_learning = False

    def answer(self, prompt: str) -> str:
        raise NotImplementedError

    def learn(self, prompt: str, answer: str) -> None:
        pass


class OllamaSystem(System):
    """Bare local model through Ollama's HTTP API (temperature 0)."""

    def __init__(self, model, host=None, timeout=300):
        self.name = f"ollama:{model}"
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout

    def answer(self, prompt):
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0, "num_ctx": 4096}}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read()).get("response", "")


class CallableSystem(System):
    """Wraps `module:function`. The module may also define `learn(prompt, answer)`."""

    def __init__(self, spec):
        mod_name, func = spec.split(":", 1)
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(mod_name)
        self.name = f"callable:{spec}"
        self._answer = getattr(mod, func)
        self._learn = getattr(mod, "learn", None)
        self.supports_learning = self._learn is not None

    def answer(self, prompt):
        return str(self._answer(prompt))

    def learn(self, prompt, answer):
        self._learn(prompt, answer)


class CommandSystem(System):
    """Runs a shell command per prompt: prompt on stdin, answer on stdout."""

    def __init__(self, command, timeout=300):
        self.name = f"cmd:{command}"
        self.argv = shlex.split(command)
        self.timeout = timeout

    def answer(self, prompt):
        p = subprocess.run(self.argv, input=prompt, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0:
            raise RuntimeError(f"command exited {p.returncode}: {p.stderr[-300:]}")
        return p.stdout


# ---- calibration systems --------------------------------------------------------------------

class TableSystem(System):
    """Knows the answer to exactly the prompts in its table; otherwise says '?'."""

    def __init__(self, table, name="table"):
        self.table, self.name = dict(table), name

    def answer(self, prompt):
        return self.table.get(prompt, "?")


class MemorizerSystem(System):
    """Learns prompt->answer by rote. Perfect recall, no generalization."""
    name = "memorizer"
    supports_learning = True

    def __init__(self):
        self.memory = {}

    def learn(self, prompt, answer):
        self.memory[prompt] = answer

    def answer(self, prompt):
        return self.memory.get(prompt, "0")


class ConstantSystem(System):
    def __init__(self, value="0"):
        self.value, self.name = value, f"constant:{value}"

    def answer(self, prompt):
        return self.value


class WrappedSystem(System):
    """A base system plus a wrapper function: wrapper(prompt, base_output) -> final output.
    A wrapper that ignores base_output models an override; one that returns it unchanged models
    a wrapper whose effect never reaches the answer."""

    def __init__(self, base, wrapper, name="wrapped"):
        self.base, self.wrapper, self.name = base, wrapper, name

    def answer(self, prompt):
        return self.wrapper(prompt, self.base.answer(prompt))


def make_system(spec):
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        return OllamaSystem(rest)
    if kind == "callable":
        return CallableSystem(rest)
    if kind == "cmd":
        return CommandSystem(rest)
    raise ValueError(f"unknown system spec {spec!r}; use ollama:<model>, callable:<module>:<func>, or cmd:<command>")
