"""Guidance for the emergency, critical and childbirth modes (the diabetes copilot has its own pipeline).

Every mode returns the same "guidance" shape so the web page can show it one way:
  headline, urgency, summary, steps[], avoid[], escalate[], ask[], notes[]
Each step cites evidence ids (E1, E2...). Ids the model invents are dropped, as in the original code.
"""
import json

from app.kb import search
from app.llm import ask
from app.modes import MODES

SHAPE = """{
  "headline": "one short sentence naming the most likely problem, or 'Unclear'",
  "urgency": "call_now" or "urgent" or "routine",
  "summary": "2 sentences: what is happening and what matters most",
  "steps": [{"text": "one clear action", "evidence": ["E1"]}],
  "avoid": [{"text": "something NOT to do", "evidence": []}],
  "escalate": [{"text": "a sign that means get more help now", "evidence": []}],
  "ask": ["a short question to ask the patient or caller right now"],
  "notes": ["anything else important that was said: age, allergies, medicines, time it started"]
}"""

RULES = """Rules:
- Use ONLY the EVIDENCE passages for medical facts, doses and time targets. Never invent a dose or an id.
- "evidence" lists ids that appear below. If no passage supports an item, give an empty list.
- Do not repeat the BUILT-IN STEPS word for word. Add what is specific to this person, or fill gaps.
- Never tell the reader to delay calling emergency services. This is support, not a diagnosis."""


def evidence_from(hits) -> list:
    return [{"id": f"E{i}", "source": h["source"], "page_label": h["page_label"], "domain": h.get("domain", ""),
             "snippet": h["text"][:300], "text": h["text"][:700], "relevant": True}
            for i, h in enumerate(hits or [], 1)]


def evidence_block(evidence) -> str:
    return "\n\n".join(f"[{e['id']}] ({e['source']} {e['page_label']})\n{e['text']}" for e in evidence) or "None."


def clean(raw: dict, evidence: list) -> dict:
    """Validate the model output: right types, only real evidence ids, short lists."""
    raw = raw if isinstance(raw, dict) else {}
    valid = {e["id"] for e in evidence}

    def items(key, limit):
        out = []
        for it in raw.get(key) or []:
            it = {"text": it, "evidence": []} if isinstance(it, str) else it
            if not isinstance(it, dict) or not str(it.get("text") or "").strip():
                continue
            cited = [str(i) for i in (it.get("evidence") if isinstance(it.get("evidence"), list) else []) if str(i) in valid]
            out.append({"text": str(it["text"]).strip(), "evidence": cited, "supported": bool(cited)})
        return out[:limit]

    strs = lambda key, n: [str(x).strip() for x in (raw.get(key) if isinstance(raw.get(key), list) else []) if str(x).strip()][:n]
    urgency = str(raw.get("urgency") or "routine").lower()
    return {"headline": str(raw.get("headline") or "").strip(), "summary": str(raw.get("summary") or "").strip(),
            "urgency": urgency if urgency in ("call_now", "urgent", "routine") else "urgent",
            "steps": items("steps", 6), "avoid": items("avoid", 5), "escalate": items("escalate", 5),
            "ask": strs("ask", 5), "notes": strs("notes", 5)}


def confidence_of(guidance: dict, evidence: list) -> str:
    steps = guidance.get("steps") or []
    if not evidence or not steps:
        return "insufficient"
    return "sufficient" if sum(s["supported"] for s in steps) * 2 >= len(steps) else "partial"


def fast_guidance(mode: str, transcript: str, tri: dict) -> tuple:
    """ONE retrieval + ONE model call. Returns (guidance, evidence, warnings)."""
    names = [p["title"] for p in tri["protocols"]]
    hits = search(mode, f"{transcript[:700]}\n{' '.join(names)}", k=6)
    warnings = []
    if hits is None:
        warnings.append("No documents are loaded for this mode yet, so only the built-in steps are shown. "
                        "Add files to the knowledge folders and run the ingest.")
        hits = []
    evidence = evidence_from(hits)
    built_in = [{"protocol": p["title"], "steps": p["steps"]} for p in tri["protocols"]]
    prompt = f"""You support {MODES[mode]['audience']} during a possible emergency. Be fast and specific.
Return ONLY a JSON object of this shape:

{SHAPE}

{RULES}
- Maximum 6 steps, most urgent first. Each under 25 words.
- If the text describes a life-threatening problem, urgency must be "call_now".

BUILT-IN STEPS ALREADY SHOWN: {json.dumps(built_in)}
EVIDENCE:
{evidence_block(evidence)}
WHAT WAS SAID:
{transcript[:4000]}"""
    guidance = clean(ask(prompt, as_json=True), evidence)
    if tri["protocols"] and guidance["urgency"] == "routine":
        guidance["urgency"] = "call_now"      # the code saw an emergency pattern: never let the model downgrade it
    return guidance, evidence, warnings


def agentic_guidance(mode: str, transcript: str, extracted: dict, evidence: list) -> dict:
    """Final step of the agentic pipeline (used by childbirth): advice built only from graded passages."""
    relevant = [e for e in evidence if e["relevant"]]
    prompt = f"""You are a clinical decision-support assistant for {MODES[mode]['audience']}.
Return ONLY a JSON object of this shape:

{SHAPE}

{RULES}
- steps: what to do or check now and at the next review. Maximum 6.
- escalate: red flags for mother or baby that need urgent senior help. Maximum 5.
- ask: what is still missing from the history.
- Everything is a suggestion for a qualified clinician to verify.

DETAILS FOUND IN THE CONVERSATION: {json.dumps(extracted)}
EVIDENCE:
{evidence_block(relevant)}
TRANSCRIPT:
{transcript}"""
    return clean(ask(prompt, as_json=True), relevant)
