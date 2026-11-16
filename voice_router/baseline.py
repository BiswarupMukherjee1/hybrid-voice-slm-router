"""Stand-in for the existing embedded voice engine: a fixed command grammar.

It understands short, well-defined commands ("set the temperature to 21 degrees",
"next song") and nothing else. Free phrasing, several requests in one sentence,
follow-ups ("a bit warmer") and questions fall outside the grammar and are returned
as not understood, which is where the hybrid router hands over to the language model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FILLERS = r"^(hey truck|ok truck|please|could you|can you|would you)\s+"
NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty one": 21, "twenty two": 22, "twenty three": 23, "twenty four": 24,
    "twenty five": 25, "twenty six": 26, "twenty seven": 27, "twenty eight": 28,
}


@dataclass
class BaselineResult:
    understood: bool
    calls: list[dict] = field(default_factory=list)
    confidence: float = 0.0


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s'-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    prev = None
    while prev != t:
        prev = t
        t = re.sub(FILLERS, "", t).strip()
    t = re.sub(r"\s+please$", "", t)
    for word, n in sorted(NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{word}\b", str(n), t)
    return t


ZONES = {"cab": "all", "interior": "all", "all": "all", "driver": "driver", "bunk": "bunk", "reading": "driver"}

_RULES: list[tuple[re.Pattern, callable]] = []


def rule(pattern: str):
    def deco(fn):
        _RULES.append((re.compile(rf"^{pattern}$"), fn))
        return fn
    return deco


@rule(r"turn (?P<state>on|off) (the )?(?P<zone>cab|interior|driver|bunk|reading|all)( )?lights?")
def _light_a(m):
    return [{"name": "set_cab_light", "arguments": {"zone": ZONES[m["zone"]], "state": m["state"]}}]


@rule(r"(?P<zone>cab|interior|driver|bunk|reading)( )?lights? (?P<state>on|off)")
def _light_b(m):
    return [{"name": "set_cab_light", "arguments": {"zone": ZONES[m["zone"]], "state": m["state"]}}]


@rule(r"set (the )?temperature to (?P<n>\d+)( degrees)?( celsius)?")
def _temp(m):
    return [{"name": "set_temperature", "arguments": {"celsius": int(m["n"])}}]


@rule(r"(set )?(the )?fan( speed| level)? (to )?(?P<n>\d)")
def _fan(m):
    return [{"name": "set_fan_speed", "arguments": {"level": int(m["n"])}}]


@rule(r"(set )?(the )?seat heating (to )?(level )?(?P<n>\d)")
def _seat(m):
    return [{"name": "set_seat_heating", "arguments": {"level": int(m["n"])}}]


@rule(r"(turn )?(the )?seat heating off")
def _seat_off(m):
    return [{"name": "set_seat_heating", "arguments": {"level": 0}}]


@rule(r"(?P<a>play|pause)( music| the music)?")
def _media_a(m):
    return [{"name": "media_control", "arguments": {"action": m["a"]}}]


@rule(r"(?P<a>next|previous) (song|track)")
def _media_b(m):
    return [{"name": "media_control", "arguments": {"action": m["a"]}}]


@rule(r"(set )?(the )?volume (to )?(?P<n>\d+)")
def _vol(m):
    return [{"name": "set_volume", "arguments": {"level": int(m["n"])}}]


@rule(r"mute")
def _mute(m):
    return [{"name": "set_volume", "arguments": {"level": 0}}]


@rule(r"call (?P<name>[a-z][a-z' -]{1,30})")
def _call(m):
    return [{"name": "call_contact", "arguments": {"name": m["name"].strip().title()}}]


@rule(r"navigate to (?P<dest>[a-z0-9][a-z0-9' -]{1,40})")
def _nav(m):
    return [{"name": "navigate_to", "arguments": {"destination": m["dest"].strip().title()}}]


class BaselineEngine:
    """Deterministic grammar: either a full match with confidence 1.0, or not understood."""

    def parse(self, text: str) -> BaselineResult:
        t = normalize(text)
        # A fixed grammar handles one command per utterance.
        body = re.sub(r"^\s*(hey|ok) truck\s*,?", "", text, flags=re.I)
        if re.search(r"\b(and|then|also)\b", t) or "," in body:
            return BaselineResult(understood=False)
        for pattern, build in _RULES:
            m = pattern.match(t)
            if m:
                calls = build(m)
                # Numbers outside the grammar's ranges are left to the validator downstream.
                return BaselineResult(understood=True, calls=calls, confidence=1.0)
        return BaselineResult(understood=False)
