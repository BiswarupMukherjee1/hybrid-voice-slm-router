"""Hybrid router: safety guard -> embedded engine -> small language model -> validator.

1. A deterministic keyword guard refuses requests for high-risk controls before any
   model runs (same answer every time, no model involved).
2. The embedded engine (fixed grammar) handles the commands it fully understands.
3. Everything else goes to the small language model, whose output is constrained to
   valid JSON over the allowed functions and then validated again.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field

from .baseline import BaselineEngine
from .slm import build_prefix, parse_output
from .validator import validate

NOT_UNDERSTOOD = "Sorry, I did not understand that."
REFUSAL = "That function is not available by voice."

GUARD_PATTERNS = [
    r"parking brake", r"hand ?brake", r"retarder", r"cruise control", r"\baxle\b",
    r"lane (keeping|assist)", r"emergency brak", r"\bunlock\b.*\bdoors?\b", r"\bopen\b.*\bdoors?\b",
    r"\b(stop|turn off|shut down|switch off) the engine\b", r"\bengine off\b", r"\bdisable\b.*\b(assist|warning|brake)",
]


def guard_hit(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in GUARD_PATTERNS)


DEFAULT_STATE = {
    "temperature_c": 20,
    "fan_level": 2,
    "seat_heating": 0,
    "volume": 12,
    "media": "paused",
    "cab_light": {"driver": "off", "bunk": "off"},
    "cab_light_brightness": 60,
}


class VehicleState:
    def __init__(self):
        self.s = copy.deepcopy(DEFAULT_STATE)

    def apply(self, call: dict) -> None:
        a = call["arguments"]
        n = call["name"]
        if n == "set_temperature":
            self.s["temperature_c"] = a["celsius"]
        elif n == "set_fan_speed":
            self.s["fan_level"] = a["level"]
        elif n == "set_seat_heating":
            self.s["seat_heating"] = a["level"]
        elif n == "set_volume":
            self.s["volume"] = a["level"]
        elif n == "media_control":
            self.s["media"] = {"play": "playing", "pause": "paused"}.get(a["action"], self.s["media"])
        elif n == "set_cab_light":
            zones = ["driver", "bunk"] if a["zone"] == "all" else [a["zone"]]
            for z in zones:
                self.s["cab_light"][z] = a["state"]
            if "brightness" in a:
                self.s["cab_light_brightness"] = a["brightness"]

    def text(self) -> str:
        return json.dumps(self.s, separators=(", ", ": "))


@dataclass
class TurnResult:
    path: str                      # guard | baseline | slm | none
    calls: list[dict]              # validated calls that would be executed
    reply: str
    latency_s: float
    rejected: list = field(default_factory=list)
    error: str | None = None
    ttft_s: float | None = None
    decode_tok_per_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: str | None = None


class Dialogue:
    """One conversation: keeps vehicle state and the turn history for follow-ups."""

    def __init__(self, mode: str, slm=None, use_guard: bool = True, max_history_turns: int = 3):
        assert mode in ("baseline", "hybrid", "slm")
        if mode != "baseline" and slm is None:
            raise ValueError("hybrid and slm modes need a language model")
        self.mode = mode
        self.slm = slm
        self.use_guard = use_guard
        self.baseline = BaselineEngine()
        self.state = VehicleState()
        self.history: list[tuple[str, str]] = []  # (driver text, assistant JSON)
        self.max_history_turns = max_history_turns

    def _messages(self, text: str) -> list[dict]:
        msgs = [{"role": "user", "content": build_prefix()}, {"role": "assistant", "content": "Ready."}]
        for user, assistant in self.history[-self.max_history_turns:]:
            msgs.append({"role": "user", "content": f"Driver: {user}"})
            msgs.append({"role": "assistant", "content": assistant})
        # The changing part (state + new request) goes last so the fixed prefix stays cacheable.
        msgs.append({"role": "user", "content": f"Vehicle state: {self.state.text()}\nDriver: {text}"})
        return msgs

    def _finish(self, text: str, result: TurnResult) -> TurnResult:
        for c in result.calls:
            self.state.apply(c)
        self.history.append((text, json.dumps({"calls": result.calls, "reply": result.reply})))
        return result

    def handle(self, text: str) -> TurnResult:
        t0 = time.perf_counter()
        if self.use_guard and self.mode != "baseline" and guard_hit(text):
            return self._finish(text, TurnResult("guard", [], REFUSAL, time.perf_counter() - t0))

        if self.mode in ("baseline", "hybrid"):
            b = self.baseline.parse(text)
            if b.understood:
                v = validate(b.calls)
                return self._finish(text, TurnResult("baseline", v.accepted, "Done.", time.perf_counter() - t0, v.rejected))
            if self.mode == "baseline":
                return self._finish(text, TurnResult("none", [], NOT_UNDERSTOOD, time.perf_counter() - t0))

        raw, stats = self.slm.generate(self._messages(text))
        calls, reply, error = parse_output(raw)
        v = validate(calls)
        return self._finish(text, TurnResult(
            "slm", v.accepted, reply or "", time.perf_counter() - t0, v.rejected, error,
            stats.ttft_s, stats.decode_tok_per_s, stats.prompt_tokens, stats.completion_tokens, raw,
        ))
