import json
from pathlib import Path

import pytest

from voice_router.baseline import BaselineEngine
from voice_router.functions import ALLOWED_FUNCTIONS, BLOCKED_FUNCTIONS, output_schema
from voice_router.router import Dialogue, guard_hit
from voice_router.scoring import calls_match, score_turn
from voice_router.slm import GenStats, parse_output
from voice_router.validator import validate

DATA = Path(__file__).resolve().parents[1] / "data" / "requests.jsonl"
CASES = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
TURNS = [t for c in CASES for t in c["turns"]]


# ---------------- validator ----------------
def test_validator_accepts_valid_and_rejects_bad():
    v = validate([
        {"name": "set_temperature", "arguments": {"celsius": 21}},
        {"name": "set_temperature", "arguments": {"celsius": 40}},          # out of range
        {"name": "release_parking_brake", "arguments": {}},                # blocked
        {"name": "make_coffee", "arguments": {}},                          # unknown
        {"name": "set_volume", "arguments": {"level": 5, "extra": 1}},     # unexpected arg
    ])
    assert len(v.accepted) == 1 and len(v.rejected) == 4
    assert v.blocked_attempts == 1


def test_schema_only_names_allowed_functions():
    names = {item["properties"]["name"]["const"] for item in output_schema()["properties"]["calls"]["items"]["anyOf"]}
    assert names == set(ALLOWED_FUNCTIONS)
    assert not names & set(BLOCKED_FUNCTIONS)


# ---------------- baseline grammar vs dataset ----------------
@pytest.mark.parametrize("turn", [t for t in TURNS if t["category"] == "simple"], ids=lambda t: t["text"])
def test_baseline_solves_every_simple_command(turn):
    r = BaselineEngine().parse(turn["text"])
    assert r.understood and calls_match(validate(r.calls).accepted, turn["expected"])


@pytest.mark.parametrize("turn", [t for t in TURNS if t["category"] in ("multi_intent", "unsafe", "question", "unsupported")],
                         ids=lambda t: t["text"])
def test_baseline_does_not_grab_complex_requests(turn):
    r = BaselineEngine().parse(turn["text"])
    assert not r.understood


def test_baseline_misses_most_free_phrasing():
    free = [t for t in TURNS if t["category"] == "free_phrasing"]
    hits = sum(BaselineEngine().parse(t["text"]).understood for t in free)
    assert hits <= len(free) // 4


# ---------------- guard ----------------
def test_guard_catches_explicit_and_misses_paraphrases():
    unsafe = [t["text"] for t in TURNS if t["category"] == "unsafe"]
    caught = [t for t in unsafe if guard_hit(t)]
    assert len(caught) >= 8                      # explicit high-risk requests
    assert len(caught) < len(unsafe)             # paraphrases are left for the model to refuse
    assert not any(guard_hit(t["text"]) for t in TURNS if t["category"] in ("simple", "free_phrasing", "multi_intent", "question"))


# ---------------- scoring ----------------
def test_scoring_ranges_contains_and_order():
    exp = [{"name": "navigate_to", "arguments": {"destination": {"contains": "malm"}}},
           {"name": "set_volume", "arguments": {"level": {"range": [13, 20]}}}]
    ok = [{"name": "set_volume", "arguments": {"level": 15}}, {"name": "navigate_to", "arguments": {"destination": "Malmö"}}]
    assert calls_match(ok, exp)
    assert not calls_match(ok[:1], exp)
    assert not calls_match([{"name": "set_volume", "arguments": {"level": 25}}, ok[1]], exp)


# ---------------- router with a fake model ----------------
class FakeSLM:
    """Returns pre-set JSON answers and records the prompts it was given."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(messages)
        return self.outputs.pop(0), GenStats(ttft_s=0.1, total_s=0.3, prompt_tokens=500, completion_tokens=20)


def test_hybrid_routes_simple_to_baseline_and_rest_to_model():
    slm = FakeSLM([json.dumps({"calls": [{"name": "set_temperature", "arguments": {"celsius": 22}}], "reply": "Warmer."})])
    d = Dialogue("hybrid", slm=slm)
    r1 = d.handle("Set the temperature to 21 degrees")
    r2 = d.handle("A bit warmer")
    assert r1.path == "baseline" and r2.path == "slm"
    assert d.state.s["temperature_c"] == 22
    last = slm.prompts[0][-1]["content"]
    assert '"temperature_c": 21' in last          # state reached the model
    assert "Set the temperature to 21" in slm.prompts[0][2]["content"]  # history reached the model


def test_guard_refuses_before_model_and_validator_blocks_bad_output():
    slm = FakeSLM([json.dumps({"calls": [{"name": "set_volume", "arguments": {"level": 99}}], "reply": "ok"})])
    d = Dialogue("hybrid", slm=slm)
    assert d.handle("Release the parking brake").path == "guard"
    r = d.handle("Make it really loud")
    assert r.calls == [] and r.rejected        # out-of-range volume never executed
    assert len(slm.prompts) == 1               # guard did not call the model


def test_parse_output_never_guesses():
    calls, reply, err = parse_output('{"calls": [ {"name": "set_vol')
    assert calls == [] and err


def test_score_unsafe_and_question():
    from voice_router.router import TurnResult
    assert score_turn("unsafe", [], TurnResult("guard", [], "That function is not available by voice.", 0))["success"]
    assert not score_turn("question", [], TurnResult("none", [], "Sorry, I did not understand that.", 0))["resolved"]


# ---------------- grammar compiles (only if llama.cpp is installed) ----------------
def test_schema_compiles_to_llama_grammar():
    lg = pytest.importorskip("llama_cpp.llama_grammar")
    gbnf = lg.json_schema_to_gbnf(json.dumps(output_schema()))
    lg.LlamaGrammar.from_string(gbnf, verbose=False)
