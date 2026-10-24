"""The four use cases ("modes") and the knowledge folders ("domains") behind them.

Add a new problem area by adding a folder under knowledge/, a DOMAINS entry here,
and listing it in the domains of one or more modes. No other code change is needed.
"""
import os

EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "112")   # national emergency number
AMBULANCE_NUMBER = os.getenv("AMBULANCE_NUMBER", "108")   # ambulance, shown as a second button

# folder name under knowledge/  ->  vector-database collection
DOMAINS = {
    "heart_stroke":        {"label": "Heart stroke",        "collection": "heart_stroke_kb"},
    "child_choking":       {"label": "Child choking",       "collection": "child_choking_kb"},
    "lifestyle_disease":   {"label": "Lifestyle disease",   "collection": "diabetes_knowledge"},   # name kept so an existing index still works
    "childbirth":          {"label": "Childbirth",          "collection": "childbirth_kb"},
    "critical_conditions": {"label": "Critical conditions", "collection": "critical_conditions_kb"},
}

# pipeline: "fast"      = instant built-in steps + ONE model call            (emergencies: no waiting)
#           "agentic"   = plan -> retrieve -> grade -> rewrite -> advise     (careful, slower)
#           "lifestyle" = the original diabetes consultation copilot
MODES = {
    "emergency": {
        "label": "Emergency",
        "short": "Stroke, choking, chest pain",
        "pipeline": "fast",
        "domains": ["heart_stroke", "child_choking", "critical_conditions"],
        "audience": "a parent, bystander or patient with no medical training. Use very short, calm, plain sentences",
        "live_voice": True,
    },
    "critical": {
        "label": "Critical care",
        "short": "Heart attack, stroke: act now",
        "pipeline": "fast",
        "domains": ["critical_conditions", "heart_stroke"],
        "audience": "a doctor, nurse or paramedic. Use concise clinical language, time targets and drug names where the evidence gives them",
        "live_voice": True,
    },
    "lifestyle": {
        "label": "Lifestyle disease",
        "short": "Diabetes and everyday conditions",
        "pipeline": "lifestyle",
        "domains": ["lifestyle_disease"],
        "audience": "a doctor in a routine consultation",
        "live_voice": False,
    },
    "maternity": {
        "label": "Childbirth",
        "short": "Labor, delivery and senior care",
        "pipeline": "agentic",
        "domains": ["childbirth"],
        "audience": ("an experienced obstetrician or midwife. Pay particular attention to advanced maternal age (35+), "
                     "older or senior patients, and any red flag for mother or baby"),
        "live_voice": False,
    },
}
DEFAULT_MODE = "lifestyle"


def case_collection(mode: str) -> str:
    """Saved consultations. 'lifestyle' keeps the original collection name so old records stay searchable."""
    return "past_consultations" if mode == "lifestyle" else f"cases_{mode}"
