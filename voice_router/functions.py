"""Vehicle function catalog.

ALLOWED functions are the only ones the language model can ever call: they are the
only names in the JSON schema used for constrained decoding, and the validator checks
them again. BLOCKED functions are high-risk truck controls that voice must never reach.
They are listed here so that requests for them can be recognised and refused.
"""

from __future__ import annotations

ALLOWED_FUNCTIONS: dict[str, dict] = {
    "set_cab_light": {
        "description": "Turn interior cab lights on or off, optionally with brightness in percent.",
        "parameters": {
            "zone": {"type": "string", "enum": ["driver", "bunk", "all"]},
            "state": {"type": "string", "enum": ["on", "off"]},
            "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["zone", "state"],
    },
    "set_temperature": {
        "description": "Set the cab climate temperature in degrees Celsius (16 to 28).",
        "parameters": {"celsius": {"type": "integer", "minimum": 16, "maximum": 28}},
        "required": ["celsius"],
    },
    "set_fan_speed": {
        "description": "Set the climate fan speed from 0 (off) to 5 (max).",
        "parameters": {"level": {"type": "integer", "minimum": 0, "maximum": 5}},
        "required": ["level"],
    },
    "set_seat_heating": {
        "description": "Set driver seat heating from 0 (off) to 3 (max).",
        "parameters": {"level": {"type": "integer", "minimum": 0, "maximum": 3}},
        "required": ["level"],
    },
    "media_control": {
        "description": "Control media playback.",
        "parameters": {"action": {"type": "string", "enum": ["play", "pause", "next", "previous"]}},
        "required": ["action"],
    },
    "set_volume": {
        "description": "Set the audio volume from 0 (mute) to 30 (max).",
        "parameters": {"level": {"type": "integer", "minimum": 0, "maximum": 30}},
        "required": ["level"],
    },
    "call_contact": {
        "description": "Start a hands-free phone call to a contact by name.",
        "parameters": {"name": {"type": "string"}},
        "required": ["name"],
    },
    "navigate_to": {
        "description": "Start navigation to a destination (city, address or place).",
        "parameters": {"destination": {"type": "string"}},
        "required": ["destination"],
    },
}

# High-risk controls: never exposed to the model, always refused.
BLOCKED_FUNCTIONS: dict[str, str] = {
    "release_parking_brake": "parking brake",
    "set_retarder": "retarder",
    "set_cruise_control": "cruise control",
    "lift_axle": "axle lift",
    "disable_lane_keeping": "lane keeping assist",
    "unlock_doors_while_driving": "door locks",
    "stop_engine": "engine",
}

MAX_CALLS = 4


def _args_schema(spec: dict) -> dict:
    return {
        "type": "object",
        "properties": spec["parameters"],
        "required": spec["required"],
        "additionalProperties": False,
    }


def output_schema() -> dict:
    """JSON schema the model must follow. `calls` comes first so the decision is
    generated before any free text (fewer tokens before the useful part)."""
    call_variants = [
        {
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": _args_schema(spec),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for name, spec in ALLOWED_FUNCTIONS.items()
    ]
    return {
        "type": "object",
        "properties": {
            "calls": {"type": "array", "items": {"anyOf": call_variants}, "maxItems": MAX_CALLS},
            "reply": {"type": "string", "maxLength": 120},
        },
        "required": ["calls", "reply"],
        "additionalProperties": False,
    }


def catalog_text() -> str:
    """Compact, human-readable function list for the prompt."""
    lines = []
    for name, spec in ALLOWED_FUNCTIONS.items():
        args = []
        for arg, a in spec["parameters"].items():
            if "enum" in a:
                t = "|".join(a["enum"])
            elif a["type"] == "integer":
                t = f"int {a.get('minimum')}-{a.get('maximum')}"
            else:
                t = a["type"]
            opt = "" if arg in spec["required"] else "?"
            args.append(f"{arg}{opt}: {t}")
        lines.append(f"- {name}({', '.join(args)}): {spec['description']}")
    return "\n".join(lines)
