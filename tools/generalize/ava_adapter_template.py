"""Template: expose your Ava-wrapped model to the harness.

I could not see your IntrextroSyn code, so I have not guessed its API. Fill in the two functions,
save the file next to your project as e.g. ava_adapter.py, then run from that folder:

    python -m generalize run --system callable:ava_adapter:answer --n 10 --out run_ava
    python -m generalize run --system ollama:gemma3:1b            --n 10 --out run_bare
    python -m generalize compare run_bare.json run_ava.json --out compare.md

Rules that make the result mean something:
  * answer() must run ONE fresh turn through the same path a real user turn takes (the live entrypoint),
    and return the text a user would see. Not an internal field, not a log line.
  * Keep the underlying model identical to the bare run (same Ollama model, same context length).
  * learn() is optional. If you define it, it should feed (prompt, correct_answer) to Ava's learning /
    outcome-memory path the way real feedback would arrive. If you leave it out, the harness skips
    the learning phase and says so.
"""


def answer(prompt: str) -> str:
    # TODO: call your wrapped system's live entrypoint and return the final user-visible reply.
    raise NotImplementedError("wire this to the Ava-wrapped entrypoint")


# def learn(prompt: str, correct_answer: str) -> None:
#     # TODO (optional): record this outcome through Ava's real learning path.
#     ...
