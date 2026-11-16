"""Small language model backend (llama.cpp through llama-cpp-python).

The model's output is constrained by a grammar generated from the JSON schema in
functions.py, so every answer is syntactically valid JSON that can only name allowed
functions. Timing is measured from the token stream: time to first token (dominated
by prompt processing) and decode speed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .functions import catalog_text, output_schema

INSTRUCTIONS = """You are the voice assistant in a truck cab. Turn the driver's request into function calls.
Rules:
- Use only the functions listed below. Use one call per requested action, several calls if the driver asks for several things.
- Relative requests use the current vehicle state: warmer/colder = 1 degree, louder/quieter = 3 volume steps, fan up/down = 1 level, brighter/dimmer = 20 percent.
- If the driver asks for anything not in the list (for example the parking brake, retarder, cruise control, axle lift, doors, engine or driver assistance), return no calls and say that it is not available by voice.
- If the driver asks a question, return no calls and answer it in one short sentence.
- "reply" is one short sentence spoken back to the driver.
Answer only with JSON: {"calls": [{"name": ..., "arguments": {...}}], "reply": "..."}

Functions:
"""


def build_prefix() -> str:
    """Fixed part of the prompt. It is identical for every request, so llama.cpp can
    reuse its KV cache and skip re-processing it (see warm vs cold time to first token)."""
    return INSTRUCTIONS + catalog_text()


@dataclass
class GenStats:
    ttft_s: float
    total_s: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def decode_tok_per_s(self) -> float:
        dt = self.total_s - self.ttft_s
        return (self.completion_tokens - 1) / dt if dt > 0 and self.completion_tokens > 1 else 0.0


class LlamaCppSLM:
    def __init__(self, model_path: str, n_threads: int = 4, n_ctx: int = 4096, max_tokens: int = 160):
        from llama_cpp import Llama  # imported here so tests run without llama.cpp

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.schema = output_schema()

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def generate(self, messages: list[dict]) -> tuple[str, GenStats]:
        t0 = time.perf_counter()
        ttft = None
        pieces: list[str] = []
        n_out = 0
        stream = self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": self.schema},
            temperature=0.0,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                pieces.append(delta)
                n_out += 1
        total = time.perf_counter() - t0
        text = "".join(pieces)
        prompt_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        return text, GenStats(ttft_s=ttft or total, total_s=total, prompt_tokens=prompt_tokens, completion_tokens=n_out)


def parse_output(text: str) -> tuple[list, str, str | None]:
    """Return (calls, reply, error). Never guesses: malformed output is an error."""
    try:
        obj = json.loads(text, strict=False)  # allow raw newlines inside the reply string
    except json.JSONDecodeError as e:
        return [], "", f"invalid JSON: {e.msg}"
    if not isinstance(obj, dict):
        return [], "", "output is not an object"
    calls = obj.get("calls", [])
    reply = obj.get("reply", "")
    return calls if isinstance(calls, list) else [], reply if isinstance(reply, str) else "", None
