"""Second safety layer: every call is checked against the allow-list and argument
ranges before anything is executed, even though constrained decoding should already
prevent invalid output. Nothing outside ALLOWED_FUNCTIONS can pass."""

from __future__ import annotations

from dataclasses import dataclass, field

from .functions import ALLOWED_FUNCTIONS, BLOCKED_FUNCTIONS, MAX_CALLS


@dataclass
class ValidationResult:
    accepted: list[dict] = field(default_factory=list)
    rejected: list[tuple[dict, str]] = field(default_factory=list)

    @property
    def blocked_attempts(self) -> int:
        return sum(1 for _, reason in self.rejected if reason.startswith("blocked"))


def _check_value(value, spec: dict) -> str | None:
    t = spec["type"]
    if t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return "not an integer"
        if "minimum" in spec and value < spec["minimum"]:
            return f"below minimum {spec['minimum']}"
        if "maximum" in spec and value > spec["maximum"]:
            return f"above maximum {spec['maximum']}"
    elif t == "string":
        if not isinstance(value, str) or not value.strip():
            return "empty or not a string"
        if "enum" in spec and value not in spec["enum"]:
            return f"not one of {spec['enum']}"
    return None


def validate_call(call: dict) -> str | None:
    """Return None if the call is valid, otherwise the reason it was rejected."""
    if not isinstance(call, dict):
        return "not an object"
    name = call.get("name")
    if name in BLOCKED_FUNCTIONS:
        return f"blocked: high-risk function {name}"
    if name not in ALLOWED_FUNCTIONS:
        return f"unknown function {name!r}"
    args = call.get("arguments")
    if not isinstance(args, dict):
        return "arguments missing"
    spec = ALLOWED_FUNCTIONS[name]
    for req in spec["required"]:
        if req not in args:
            return f"missing argument {req}"
    for key, value in args.items():
        if key not in spec["parameters"]:
            return f"unexpected argument {key}"
        problem = _check_value(value, spec["parameters"][key])
        if problem:
            return f"{key}: {problem}"
    return None


def validate(calls) -> ValidationResult:
    result = ValidationResult()
    if not isinstance(calls, list):
        result.rejected.append(({"calls": calls}, "calls is not a list"))
        return result
    for i, call in enumerate(calls):
        if i >= MAX_CALLS:
            result.rejected.append((call, "too many calls"))
            continue
        reason = validate_call(call)
        if reason is None:
            result.accepted.append(call)
        else:
            result.rejected.append((call, reason))
    return result
