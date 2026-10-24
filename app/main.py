"""Consultation Copilot: FastAPI server.

Four use cases (see app/modes.py):
  emergency  stroke, child choking, chest pain  -> instant built-in steps + one model call
  critical   heart attack / stroke for doctors  -> instant clinician checklist + one model call
  lifestyle  diabetes and everyday conditions   -> the original agentic consultation copilot
  maternity  childbirth, labor and delivery     -> agentic retrieval over the childbirth folder
Every mode accepts typed/pasted conversation text or a recorded voice conversation.
"""
import asyncio
import base64
import io
import json
import os
import re
import secrets
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

from app.db import USE_POSTGRES, db, init_schema   # noqa: E402
from app.guidance import agentic_guidance, confidence_of, fast_guidance   # noqa: E402
from app.kb import RETRIEVAL_K, case_stores, ingest_all, kb_status, mode_chunks   # noqa: E402
from app.kb import search as kb_search   # noqa: E402
from app.llm import LLMS, ask   # noqa: E402
from app.modes import AMBULANCE_NUMBER, DEFAULT_MODE, EMERGENCY_NUMBER, MODES   # noqa: E402
from app.triage import all_protocols, split_dialogue, triage   # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", BASE_DIR / "frontend"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR / "static"))   # holds the built tailwind.css
SQLITE_PATH = os.getenv("SQLITE_PATH", str(BASE_DIR / "records.db"))   # patient records (the permanent record book)
STT_LANG_HINT = os.getenv("STT_LANG_HINT", "English")   # e.g. Hindi, Kannada, or "auto"
MAX_ROUNDS = int(os.getenv("MAX_RETRIEVAL_ROUNDS", "2"))          # 1 = never rewrite a weak query, 2 = one rewrite
FAST_TIMEOUT = float(os.getenv("EMERGENCY_LLM_TIMEOUT", "25"))    # seconds the emergency modes wait for the model
APP_USER = os.getenv("APP_USER", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")                      # empty = no login. Set it before exposing the app
TOOLS = ("guidelines", "clinic_records")
Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Patient records (Supabase Postgres, or SQLite when DATABASE_URL is empty)
# ============================================================


init_schema()   # Supabase (DATABASE_URL set) or local SQLite; see app/db.py


def _json_or(text, default):
    try:
        value = json.loads(text) if isinstance(text, str) else text
        return value if isinstance(value, type(default)) else default
    except (json.JSONDecodeError, TypeError):
        return default


def row_to_dict(row) -> dict:
    d = dict(row)
    d["medicines"] = _json_or(d.get("medicines") or "[]", [])
    d["dialogue"] = _json_or(d.get("dialogue") or "[]", [])
    d["sheet"] = _json_or(d.get("sheet") or "{}", {})
    d["prediction"] = _json_or(d.get("prediction") or "{}", {})
    return d


def slim(rec: dict) -> dict:
    """Only what the web page needs for the 'earlier visits' / 'similar cases' cards."""
    out = {k: rec.get(k) for k in ("id", "created_at", "patient_name", "diagnosis", "medicines")}
    out["doctor_approach"] = (rec.get("doctor_approach") or "")[:400]
    return out


# ============================================================
# 4. Speech-to-text: Gemini (same GOOGLE_API_KEY, no local model)
# ============================================================

_genai_client = None
STT_MODELS = [m.strip() for m in os.getenv("STT_MODEL", os.getenv("GEMINI_MODELS", "gemini-3.5-flash")).split(",") if m.strip()]
STT_INLINE_LIMIT = 15 * 1024 * 1024   # larger recordings go through the Files API


def _stt_prompt() -> str:
    hint = "" if STT_LANG_HINT.lower() == "auto" else f" The speech is in {STT_LANG_HINT}."
    return ("Transcribe this audio exactly as spoken. If more than one person speaks, start each turn with "
            "'Doctor:' / 'Patient:' (or 'Speaker 1:' / 'Speaker 2:' if unclear). Keep medical terms, drug names, "
            "doses and numbers exact. If there is no speech, return nothing. Return only the transcript." + hint)


def transcribe(path: str) -> str:
    """Audio file (mono 16 kHz WAV from the web page) -> text, using Gemini."""
    global _genai_client
    from google import genai
    from google.genai import types
    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    client = _genai_client

    if os.path.getsize(path) > STT_INLINE_LIMIT:
        audio = client.files.upload(file=path, config={"mime_type": "audio/wav"})
        while audio.state and audio.state.name == "PROCESSING":
            time.sleep(2)
            audio = client.files.get(name=audio.name)
    else:
        with open(path, "rb") as f:
            audio = types.Part.from_bytes(data=f.read(), mime_type="audio/wav")

    last_error = None
    for model in STT_MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=[_stt_prompt(), audio])
            return (resp.text or "").strip()
        except Exception as error:
            print(f"[stt] {model} failed: {type(error).__name__}: {str(error)[:200]}")
            last_error = error
    raise RuntimeError(f"Speech-to-text failed on every Gemini model: {last_error}")


# ============================================================
# 5. Diabetes screening knowledge (plain code, no LLM)
# ============================================================

TYPE_LABELS = {
    "t2": "Type 2 diabetes",
    "t1": "Type 1 diabetes",
    "pre": "Prediabetes",
    "gdm": "Gestational diabetes",
    "lada": "LADA (latent autoimmune diabetes in adults)",
    "mody": "MODY (monogenic diabetes)",
    "sec": "Secondary diabetes (drug, pancreatic or endocrine cause)",
    "und": "Diabetes, type to be determined",
    "other": "Another cause (diabetes unlikely)",
}

# ADA diagnostic thresholds, shown to the doctor as a quick reference.
DIAGNOSTIC_CRITERIA = [
    {"test": "HbA1c", "diabetes": "6.5% or more", "prediabetes": "5.7 to 6.4%"},
    {"test": "Fasting plasma glucose (no food for 8 hours)", "diabetes": "126 mg/dL (7.0 mmol/L) or more",
     "prediabetes": "100 to 125 mg/dL (5.6 to 6.9 mmol/L)"},
    {"test": "2-hour OGTT (75 g glucose)", "diabetes": "200 mg/dL (11.1 mmol/L) or more",
     "prediabetes": "140 to 199 mg/dL (7.8 to 11.0 mmol/L)"},
    {"test": "Random plasma glucose with classic symptoms", "diabetes": "200 mg/dL (11.1 mmol/L) or more",
     "prediabetes": "Not applicable"},
]
CRITERIA_NOTE = ("Without clear symptoms, a positive result should be confirmed by repeating the test. "
                 "Based on ADA criteria; follow your local guideline where it differs.")

BASE_TESTS = {
    "common": [
        ("HbA1c", "Average blood sugar over about 3 months. Diabetes at 6.5% or more."),
        ("Fasting plasma glucose (8 hour fast)", "Diabetes at 126 mg/dL (7.0 mmol/L) or more."),
        ("Random plasma glucose", "200 mg/dL (11.1 mmol/L) or more with classic symptoms supports diabetes."),
    ],
    "t2": [
        ("2-hour OGTT (75 g glucose)", "Settles the diagnosis when HbA1c or fasting glucose is borderline."),
        ("Lipid profile", "Baseline cardiovascular risk."),
        ("Serum creatinine / eGFR and urine albumin-creatinine ratio", "Baseline kidney check."),
    ],
    "t1": [
        ("Blood or urine ketones", "Looks for ketosis, which needs urgent care."),
        ("Islet autoantibodies (GAD65, IA-2, ZnT8)", "Supports autoimmune (type 1) diabetes."),
        ("C-peptide", "Shows how much insulin the body still makes."),
    ],
    "lada": [
        ("GAD65 antibodies", "Positive in most autoimmune diabetes."),
        ("C-peptide", "Low or falling levels suggest insulin deficiency."),
    ],
    "mody": [
        ("Genetic testing for MODY genes", "Considered with strong family history and early onset."),
        ("C-peptide", "Usually preserved in MODY."),
    ],
    "gdm": [
        ("75 g OGTT at 24 to 28 weeks of pregnancy", "Standard screening for gestational diabetes."),
    ],
    "pre": [
        ("2-hour OGTT (75 g glucose)", "Separates prediabetes from diabetes."),
        ("Repeat HbA1c or fasting glucose in 3 to 6 months", "Tracks progression."),
    ],
    "sec": [
        ("Review of steroid and other drug use", "Some medicines raise blood sugar."),
        ("Pancreatic and endocrine work-up as clinically indicated", "Looks for a secondary cause."),
    ],
}

SYMPTOM_PATTERNS = {
    "Frequent urination": [r"polyuria", r"urinat\w* (a lot|often|frequent\w*|more)", r"frequent (urination|urine)",
                           r"(pee|peeing|toilet|bathroom)\b.{0,30}(often|a lot|night|frequent\w*|more)",
                           r"getting up at night to (pass|urinate|pee)"],
    "Excess thirst": [r"polydipsia", r"(very|always|extremely|so|constantly) thirsty", r"thirst\w*",
                      r"drinking (a lot|too much|more) (of )?water"],
    "Increased hunger": [r"polyphagia", r"(always|very|extremely|constantly) hungry", r"hungry all the time"],
    "Unexplained weight loss": [r"los(t|ing|e) (a lot of )?weight", r"weight loss", r"lost \d+ ?(kg|kilos|pounds|lbs)"],
    "Tiredness": [r"fatigue", r"tired", r"weakness", r"no energy", r"exhausted"],
    "Blurred vision": [r"blur\w*", r"vision (is|has been|got) (worse|poor|fuzzy)", r"can'?t see (clearly|well)"],
    "Slow-healing wounds or frequent infections": [r"wounds? .{0,30}(heal|slow)", r"slow(ly)? heal\w*",
                                                   r"(frequent|recurrent|repeated) infections?", r"boils?", r"fungal"],
    "Tingling or numbness": [r"tingling", r"numb\w*", r"pins and needles", r"burning (in|sensation).{0,20}(feet|foot|hands)"],
    "Dry mouth or itching": [r"dry mouth", r"itch\w*"],
}

RISK_PATTERNS = {
    "Family history of diabetes": r"(father|mother|parents?|brother|sister|family|grand\w+|uncle|aunt)\b.{0,40}(diabet\w*|sugar)",
    "Overweight or weight gain": r"overweight|obes\w*|gained weight|weight gain|\bbmi\b",
    "Low physical activity": r"sedentary|no exercise|don'?t exercise|desk job|little (physical )?activity",
    "Sugary or processed diet": r"sweets?|sugary|soft drinks?|cold drinks?|\bsodas?\b|fast food|junk",
    "High blood pressure": r"hypertens\w*|high blood pressure|\bbp\b",
    "Past gestational diabetes or PCOS": r"gestational|pcos|polycystic",
    "Smoking or alcohol": r"smok\w*|alcohol|drinks? (beer|whisky|wine)",
}

RED_FLAG_PATTERNS = [
    (r"fruity (smell|breath)|breath smells? (of )?(fruit|acetone)", "fruity breath", True),
    (r"(deep|rapid|fast|laboured|labored) breath\w*", "rapid or deep breathing", True),
    (r"confus\w*|drowsy|unconscious|faint\w*", "confusion, drowsiness or fainting", True),
    (r"vomit\w*", "vomiting", False),
    (r"(abdominal|stomach|belly) pain", "abdominal pain", False),
]

NEGATION_RE = re.compile(r"(\bno\b|\bnot\b|\bnever\b|n't|\bdenies\b|\bwithout\b|\bnone\b)[^.?!]{0,25}$")


def _found(patterns, text: str) -> bool:
    """True when a pattern matches and is not negated just before the match ("no weight loss")."""
    for pat in patterns if isinstance(patterns, list) else [patterns]:
        for m in re.finditer(pat, text):
            if not NEGATION_RE.search(text[max(0, m.start() - 30):m.start()]):
                return True
    return False


def rule_screen(transcript: str) -> dict:
    t = transcript.lower()
    symptoms = [name for name, pats in SYMPTOM_PATTERNS.items() if _found(pats, t)]
    risks = [name for name, pat in RISK_PATTERNS.items() if _found(pat, t)]
    flags = [(label, strong) for pat, label, strong in RED_FLAG_PATTERNS if _found(pat, t)]
    red = []
    if any(strong for _, strong in flags) or len(flags) >= 2:
        red.append("Warning signs that can point to diabetic ketoacidosis were mentioned ("
                   + ", ".join(label for label, _ in flags)
                   + "). Consider a same-day blood glucose and ketone check.")
    return {"symptoms": symptoms, "risks": risks, "red_flags": red}


def type_key(name: str) -> str:
    """Map any wording of a diagnosis ('Type II DM', 'NIDDM', 'GDM', 'steroid-induced'...) to one type key."""
    n = re.sub(r"[^a-z0-9. ]+", " ", str(name or "").lower())
    n = re.sub(r"\s+", " ", n).strip()
    if not n or n.startswith(("not mentioned", "not stated", "none", "unclear")):
        return "other"
    if "another cause" in n or "unlikely" in n:
        return "other"
    if "lada" in n or "latent" in n or "1.5" in n:
        return "lada"
    if "mody" in n or "monogenic" in n:
        return "mody"
    if "gestation" in n or re.search(r"\bgdm\b", n):
        return "gdm"
    if "non insulin" in n or re.search(r"\b(type (2|ii|two)|t2d\w*|niddm|dm ?2|adult onset)\b", n):
        return "t2"
    if "insulin dependent" in n or "juvenile" in n or re.search(r"\b(type (1|i|one)|t1d\w*|iddm|dm ?1)\b", n):
        return "t1"
    if "prediab" in n or "pre diab" in n or "impaired" in n or "borderline" in n or re.search(r"\b(igt|ifg)\b", n):
        return "pre"
    if any(w in n for w in ("secondary", "steroid", "induced", "pancrea", "cushing", "cystic fibrosis", "hemochromatosis")):
        return "sec"
    if "determined" in n:
        return "und"
    if "diabet" in n or n == "dm" or "sugar" in n:
        return "und"
    return "other"


TEST_TOKENS = [("hba1c", "hba1c"), ("fasting", "fasting"), ("glucose tolerance", "ogtt"), ("ogtt", "ogtt"),
               ("random", "random"), ("peptide", "peptide"), ("ketone", "ketone"), ("autoantibod", "antibody"),
               ("gad65", "antibody"), ("lipid", "lipid"), ("creatinine", "kidney"), ("egfr", "kidney")]


def _test_key(name: str) -> str:
    n = name.lower()
    for token, key in TEST_TOKENS:
        if token in n:
            return key
    return re.sub(r"[^a-z0-9]", "", n)[:8]


def _words(s: str) -> set:
    return set(re.findall(r"[a-z]{3,}", str(s).lower()))


def _heard(quote: str, transcript: str) -> bool:
    q = _words(quote)
    return bool(q) and len(q & _words(transcript)) / len(q) >= 0.7


def _strs(value, limit: int = 6) -> list:
    return [str(x).strip() for x in (value if isinstance(value, list) else []) if str(x).strip()][:limit]


AGE_RE = re.compile(r"\b(?:i am|i'm|im|age|aged)\s*:?\s*(\d{1,2})\b(?!\s*(?:kg|mg|weeks?|days?|months?|units?|%|times))"
                    r"|\b(\d{1,2})[- ]?(?:years?|yrs?)[- ]?old\b|\b(\d{1,2})\s*y/?o\b")
RELATIVES = ("father", "mother", "grandfather", "grandmother", "aunt", "uncle", "brother", "sister", "sibling", "cousin", "parents")
EXPLICIT_RE = re.compile(r"\b(type[\s-]*(?:1\.5|1|2|ii|i|one|two)|t[12]dm|niddm|iddm|gestational diabetes|prediab\w*|pre-diabetes|lada|mody|secondary diabetes)\b")


def _age(t: str):
    m = AGE_RE.search(t)
    return next((int(g) for g in m.groups() if g), None) if m else None


def _lab(t: str, pattern: str):
    m = re.search(pattern + r"\D{0,25}?(\d+(?:\.\d+)?)", t)
    return float(m.group(1)) if m else None


def rule_types(transcript: str, rules: dict) -> list:
    """Plain-code ranking of diabetes types, used only when the model gave none. Every line says why."""
    t = transcript.lower()
    age = _age(t)
    sym, risks, flags = set(rules["symptoms"]), set(rules["risks"]), bool(rules["red_flags"])
    fasting = _lab(t, r"(?:fasting|fbs|fpg)")
    a1c = _lab(t, r"hba1c|a1c")
    high_sugar = (fasting is not None and fasting >= 126) or (a1c is not None and a1c >= 6.5)
    pre_range = (fasting is not None and 100 <= fasting < 126) or (a1c is not None and 5.7 <= a1c < 6.5)
    relatives = sum(1 for r in RELATIVES if re.search(rf"\b{r}\b.{{0,40}}(diabet|sugar)", t))
    weight_loss = "Unexplained weight loss" in sym
    lean = _found(r"\b(slim|lean|thin|underweight)\b", t)
    over = "Overweight or weight gain" in risks
    family = "Family history of diabetes" in risks
    found, order = {}, []

    def add(key, like, why):
        if key not in found:
            found[key] = {"key": key, "type": TYPE_LABELS[key], "likelihood": like, "reasons": [], "against": []}
            order.append(key)
        found[key]["reasons"].append(why)

    for m in EXPLICIT_RE.finditer(t):
        if not NEGATION_RE.search(t[max(0, m.start() - 30):m.start()]):
            k = type_key(m.group(1))
            if k in TYPE_LABELS and k not in ("other", "und"):
                add(k, "high", "Named in the conversation")
    if _found(r"pregnan|gestation|antenatal|weeks along", t):
        add("gdm", "high" if sym or high_sugar else "moderate", "Pregnancy mentioned")
    if _found(r"steroid|prednis|dexameth|hydrocortisone|corticoster|pancreatitis|cushing|hemochromatosis|haemochromatosis|cystic fibrosis", t):
        add("sec", "high" if sym else "moderate", "A medicine or condition that can raise blood sugar was mentioned")
    if age is not None and age < 30 and (weight_loss or flags):
        add("t1", "high" if flags else "moderate", f"Young patient ({age}) with " + ("warning signs of ketoacidosis" if flags else "weight loss"))
    if lean and weight_loss and (age is None or 30 <= age <= 70):
        add("lada", "moderate", "Adult who is lean with weight loss: autoimmune diabetes is possible")
    if relatives >= 2 and (age is None or age < 35):
        add("mody", "moderate", "Several relatives with diabetes and young age")
    if (age is None or age >= 30) and (over or family or (age or 0) >= 40) and (len(sym) >= 1 or len(risks) >= 2 or high_sugar):
        add("t2", "high" if (len(sym) >= 2 and (over or family)) or high_sugar else "moderate",
            "Adult with " + ", ".join(x for x in (["overweight or weight gain"] if over else []) + (["family history"] if family else [])
                                      + ([f"age {age}"] if age and age >= 40 else []) + (list(sym)[:2] if sym else []))[:140])
    if pre_range or (not sym and risks):
        add("pre", "high" if pre_range else "moderate",
            "Blood sugar in the prediabetes range" if pre_range else "Risk factors without symptoms")
    rank = {"high": 0, "moderate": 1, "low": 2}
    return sorted((found[k] for k in order), key=lambda x: rank[x["likelihood"]])[:3]


def fallback_questions(transcript: str, rules: dict) -> list:
    """What the doctor should still ask, worked out from what the conversation did NOT cover."""
    t = transcript.lower()
    asks = []
    if not rules["symptoms"]:
        asks.append("Ask about thirst, passing urine, hunger, weight change, tiredness and blurred vision.")
    if _age(t) is None:
        asks.append("Ask the patient's age, and about pregnancy if relevant.")
    if "Family history of diabetes" not in rules["risks"]:
        asks.append("Ask about diabetes in parents, brothers, sisters and grandparents.")
    if not re.search(r"steroid|prednis|medicine|tablet|drug", t):
        asks.append("Ask about current medicines, especially steroids.")
    if not re.search(r"hba1c|fasting|blood sugar|sugar level|glucose|report", t):
        asks.append("Ask for any earlier blood sugar or HbA1c reports.")
    return asks[:4]


LIKELIHOOD_RANK = {"high": 0, "moderate": 1, "low": 2}


def build_prediction(data: dict, rules: dict, transcript: str) -> dict:
    """Validate the model's screening output and combine it with the keyword check."""
    data = data if isinstance(data, dict) else {}

    symptoms = []
    for item in data.get("symptoms") or []:
        if isinstance(item, str):
            item = {"symptom": item, "quote": ""}
        if not isinstance(item, dict) or not str(item.get("symptom") or "").strip():
            continue
        quote = str(item.get("quote") or "").strip()
        symptoms.append({"symptom": str(item["symptom"]).strip(), "quote": quote,
                         "heard": _heard(quote, transcript) if quote else None, "source": "ai"})
    if not symptoms:   # model gave nothing: fall back to the keyword check
        symptoms = [{"symptom": s, "quote": "", "heard": True, "source": "keyword"} for s in rules["symptoms"]]

    routine_in = data.get("routine") if isinstance(data.get("routine"), dict) else {}
    routine = {k: str(routine_in.get(k) or "Not mentioned").strip() or "Not mentioned"
               for k in ("diet", "activity", "sleep", "family_history", "other")}
    known = sum(1 for v in routine.values() if not v.lower().startswith("not mentioned"))

    n_sym = max(len(symptoms), len(rules["symptoms"]))
    if bool(data.get("ready")) and n_sym >= 2:
        status = "complete"
    elif n_sym >= 1 or known >= 2 or len(rules["risks"]) >= 1:
        status = "preliminary"
    else:
        status = "insufficient"

    types = []
    for item in data.get("possible_types") or []:
        if not isinstance(item, dict):
            continue
        key = type_key(item.get("type"))
        like = str(item.get("likelihood") or "low").lower()
        types.append({"key": key, "type": TYPE_LABELS[key], "likelihood": like if like in LIKELIHOOD_RANK else "low",
                      "reasons": _strs(item.get("reasons"), 4), "against": _strs(item.get("against"), 3)})
    seen, unique = set(), []
    for t in sorted(types, key=lambda x: LIKELIHOOD_RANK[x["likelihood"]]):
        if t["key"] not in seen:
            seen.add(t["key"])
            unique.append(t)
    types = unique[:4]
    if status == "insufficient":
        types = []
    elif not types:
        types = rule_types(transcript, rules)
    if status != "insufficient" and not types and n_sym >= 2:
        types = [{"key": "und", "type": TYPE_LABELS["und"], "likelihood": "moderate" if n_sym >= 3 else "low",
                  "reasons": [s["symptom"] for s in symptoms][:4], "against": []}]

    tests, seen_tests = [], set()

    def add_test(name, why, source):
        key = _test_key(name)
        if name and key not in seen_tests:
            seen_tests.add(key)
            tests.append({"test": name, "why": why, "source": source})

    for item in data.get("tests") or []:
        if isinstance(item, dict):
            add_test(str(item.get("test") or "").strip(), str(item.get("why") or "").strip(), "agent")
    if status != "insufficient":
        for name, why in BASE_TESTS["common"]:
            add_test(name, why, "standard")
        for t in types[:2]:
            for name, why in BASE_TESTS.get(t["key"], []):
                add_test(name, why, "standard")

    return {
        "status": status,
        "summary": str(data.get("summary") or "").strip(),
        "symptoms": symptoms,
        "routine": routine,
        "risk_factors": rules["risks"],
        "possible_types": types,
        "tests": tests[:14],
        "missing_info": _strs(data.get("missing_info"), 6) or fallback_questions(transcript, rules),
        "red_flags": rules["red_flags"] + [r for r in _strs(data.get("red_flags"), 3) if r not in rules["red_flags"]],
        "criteria": DIAGNOSTIC_CRITERIA,
        "criteria_note": CRITERIA_NOTE,
    }


# ============================================================
# 6. Agentic RAG pipeline (LangGraph)
#
#   transcribe -> extract -> predict -> sheet -> history -> plan -> retrieve -> advise -> finalize
#
#   predict   : screens the patient's symptoms and routine for possible diabetes types and the tests that confirm them
#   history   : earlier visits, similar patients and the doctor's prescribing trends from the clinic's own records
#   plan      : the LLM writes 2-3 search questions and picks a tool for each
#   retrieve  : search, grade every passage for relevance, rewrite weak queries, search again
#   advise    : suggestions may only cite passages that passed grading
#   finalize  : confidence level, warnings and metrics for the web page
# ============================================================


class AgentState(TypedDict, total=False):
    mode: str
    guidance: dict
    audio_path: str
    patient_name: str
    transcript: str
    extracted: dict
    prediction: dict
    sheet: dict
    sheet_warnings: list
    patient_history: list
    similar_cases: list
    trends: dict
    questions: list
    evidence: list
    rounds: int
    steps: list
    warnings: list
    advice: dict
    confidence: str
    agent: dict


def transcribe_node(state: AgentState) -> dict:
    if state.get("transcript"):   # typed or pasted text: nothing to transcribe
        return {}
    text = transcribe(state["audio_path"])
    if not text:
        raise ValueError("No speech was detected in the recording.")
    return {"transcript": text}


DRUG_RE = re.compile(
    r"\b(metformin|glimepiride|gliclazide|glipizide|glibenclamide|glyburide|insulin|glargine|detemir|degludec|lispro|aspart|glulisine|"
    r"nph|mixtard|sitagliptin|vildagliptin|linagliptin|saxagliptin|teneligliptin|empagliflozin|dapagliflozin|canagliflozin|"
    r"liraglutide|semaglutide|dulaglutide|exenatide|tirzepatide|pioglitazone|acarbose|voglibose|repaglinide|nateglinide|"
    r"atorvastatin|rosuvastatin|amlodipine|telmisartan|losartan|ramipril)\b", re.I)
DX_RE = re.compile(r"\b(?:type[\s-]*(?:1\.5|1|2|ii|i|one|two)(?:\s+diabetes(?:\s+mellitus)?|\s*dm)?|t[12]dm|niddm|iddm|gestational diabetes|gdm|"
                   r"pre-?diabetes|lada|mody(?:\s*\d)?|secondary diabetes|steroid[- ]induced (?:diabetes|hyperglycemia)|diabetes mellitus)\b", re.I)
STOP_RE = re.compile(r"\b(?:stop|avoid|do not|don't|not|no|rather than|instead of|discontinue)\b", re.I)


def _doctor_text(transcript: str) -> str:
    turns = split_dialogue(transcript)
    doctor = [t["text"] for t in turns if t["speaker"] == "Doctor"]
    return " ".join(doctor) if doctor else transcript


def fallback_diagnosis(transcript: str) -> str:
    text = _doctor_text(transcript)
    for m in DX_RE.finditer(text):
        before = text[max(0, m.start() - 35):m.start()].lower()
        if NEGATION_RE.search(before) or re.search(r"rule[sd]? out|unlikely|exclude", before):
            continue
        return m.group(0).strip()
    return "Not mentioned"


def fallback_medicines(transcript: str) -> list:
    text, meds, seen = _doctor_text(transcript), [], set()
    for m in DRUG_RE.finditer(text):
        key = m.group(0).lower()
        if key in seen:
            continue
        seen.add(key)
        before = re.split(r"[.?!;]\s", text[max(0, m.start() - 60):m.start()])[-1]   # only the same sentence counts
        tail = text[m.end():m.end() + 70]
        dose = re.search(r"\d+(?:\.\d+)?\s?(?:mg|mcg|g|units?|iu)\b[^.?!;]{0,35}", tail, re.I)
        dose_text = re.split(r"\s(?:and|then|but)\s", dose.group(0))[0].strip(" ,") if dose else "Not mentioned"
        given = re.search(r"start|prescrib|take|continue|add|begin|give|increase|switch|need", before + " " + tail[:20], re.I)
        meds.append({"name": m.group(0).capitalize(), "dosage": dose_text,
                     "status": "prescribed" if given and not STOP_RE.search(before) else "discussed"})
    return meds


def extract_node(state: AgentState) -> dict:
    prompt = f"""You are an information extraction assistant for a clinic.
Read this doctor-patient conversation transcript and return ONLY a JSON object:

{{
  "dialogue": [{{"speaker": "Doctor" or "Patient", "text": "..."}}],
  "diagnosis": "diagnosis the doctor stated, or 'Not mentioned'",
  "medicines": [{{"name": "...", "dosage": "as spoken or 'Not mentioned'", "status": "prescribed" or "discussed"}}],
  "vitals": "vitals explicitly mentioned, or 'Not mentioned'",
  "doctor_approach": "2-4 sentences: how the doctor assessed the problem and the steps taken to solve it"
}}

Rules: never invent a diagnosis, medicine, dosage or vital. Keep medicine names and doses as spoken.
Split the transcript into turns and label speakers from context. No medical advice.

TRANSCRIPT:
{state["transcript"]}"""
    warnings = list(state.get("warnings", []))
    try:
        data = ask(prompt, as_json=True)
    except RuntimeError as error:
        print(f"[agent] extraction failed, using keyword rules: {error}")
        data = {}
        warnings.append("The language model was unavailable, so the diagnosis and medicines were read with keyword rules only. Check them.")
    data = data if isinstance(data, dict) else {}
    transcript = state["transcript"]
    if not data.get("dialogue"):
        data["dialogue"] = split_dialogue(transcript)
    if not str(data.get("diagnosis") or "").strip():
        data["diagnosis"] = fallback_diagnosis(transcript)
    if not isinstance(data.get("medicines"), list):        # a key the model left out; an empty list it gave on purpose is kept
        data["medicines"] = fallback_medicines(transcript)
    data.setdefault("vitals", "Not mentioned")
    data.setdefault("doctor_approach", "")
    return {"extracted": data, "warnings": warnings}


def predict_node(state: AgentState) -> dict:
    """Screen what the patient said (symptoms, routine, family history) for possible diabetes types."""
    transcript = state["transcript"]
    rules = rule_screen(transcript)
    labels = [v for k, v in TYPE_LABELS.items() if k != "und"]
    prompt = f"""You are a diabetes screening assistant helping a doctor. Read the consultation and return ONLY a JSON object:

{{
  "ready": true or false,
  "summary": "2 sentences: what the patient described and what it suggests",
  "symptoms": [{{"symptom": "short name", "quote": "the patient's own words, under 15 words"}}],
  "routine": {{"diet": "...", "activity": "...", "sleep": "...", "family_history": "...", "other": "age, weight, pregnancy, medicines like steroids, if said"}},
  "possible_types": [{{"type": "one of {json.dumps(labels)}", "likelihood": "high" or "moderate" or "low",
                      "reasons": ["why it fits"], "against": ["what does not fit, or what is unknown"]}}],
  "tests": [{{"test": "...", "why": "what it would confirm or rule out"}}],
  "missing_info": ["questions the doctor should still ask the patient"],
  "red_flags": ["urgent warning signs, if any"]
}}

Rules:
- Use ONLY what is in the transcript. Never invent a symptom, number or history. Write "Not mentioned" for unknown routine details.
- "ready" is true only when the patient has described their symptoms and/or daily routine in enough detail to screen.
- Order possible_types from most to least likely. At most 4. Use an empty list if there is not enough information.
- Typical patterns: type 1 = young, sudden onset, weight loss, thirst, ketones; type 2 = adult, overweight, gradual onset,
  family history, inactivity; prediabetes = mild or no symptoms with risk factors; gestational = pregnancy;
  LADA = adult with slow onset but lean or antibody-type features; MODY = strong family history across generations, young onset.
- Tests are for the doctor to order to CONFIRM or rule out the possibilities. This is screening, not a diagnosis.

KEYWORD HINTS (may be wrong, transcript wins): {json.dumps(rules)}

TRANSCRIPT:
{transcript}"""
    try:
        data = ask(prompt, as_json=True)
    except RuntimeError as error:
        print(f"[agent] prediction failed, using keyword check only: {error}")
        data = {}
    return {"prediction": build_prediction(data, rules, transcript)}


def check_sheet(sheet: dict, transcript: str) -> list:
    """Plain-code cross-check of the take-home sheet against the transcript (no LLM)."""
    t = transcript.lower()
    warns = []
    for m in sheet.get("medicines") or []:
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        first = re.split(r"[\s/(,\-]+", name.lower())[0]
        if first and first not in t:
            warns.append(f"Medicine \"{name}\" was not clearly heard in the recording.")
        dose = str(m.get("dose") or "")
        numbers = re.findall(r"\d+(?:\.\d+)?", dose)
        missing = [n for n in numbers if not re.search(rf"(?<![\d.]){re.escape(n)}(?!\d)", t)]
        if missing:
            warns.append(f"Dose \"{dose}\" for {name} could not be found in the transcript. Check it.")
    dx = str(sheet.get("diagnosis") or "").lower()
    words = [w for w in re.findall(r"[a-z]{4,}", dx) if w not in ("stated", "not")]
    if words and not any(w in t for w in words):
        warns.append("The diagnosis on the sheet does not appear in the transcript. Check it.")
    return warns


def sheet_node(state: AgentState) -> dict:
    """A patient take-home sheet built ONLY from what the doctor said."""
    prompt = f"""You are preparing a take-home care sheet for a patient, based strictly on this consultation.
Return ONLY a JSON object:

{{
  "diagnosis": "diagnosis the doctor stated, or 'Not stated'",
  "summary": "2-3 sentences in very simple language telling the patient what the doctor found and the plan",
  "medicines": [{{"name": "...", "dose": "...", "how_to_take": "when/how often/with food etc.", "duration": "..."}}],
  "lifestyle": {{"diet": [], "exercise": [], "monitoring": [], "other": []}},
  "follow_up": "when to return or what tests the doctor asked for, or 'Not stated'"
}}

Rules:
- Include ONLY what the doctor actually said. Never add your own medicines, doses, foods or advice.
- If the doctor did not give a detail (dose, duration...), write "Ask your doctor".
- Only list medicines the doctor prescribed or told the patient to take; skip ones that were merely discussed and rejected.
- Lifestyle items are short, plain sentences the patient can follow (e.g. "Walk for 30 minutes every day").
- Use empty lists when the doctor gave no advice in that category.

TRANSCRIPT:
{state["transcript"]}

HELPER (may contain errors, transcript wins):
{json.dumps(state["extracted"])}"""
    failed = False
    try:
        sheet = ask(prompt, as_json=True)
    except RuntimeError as error:
        print(f"[agent] sheet failed, building it from the extracted details: {error}")
        sheet, failed = {}, True
    sheet = sheet if isinstance(sheet, dict) else {}
    ex = state["extracted"]
    if failed:
        dx = str(ex.get("diagnosis") or "")
        sheet["diagnosis"] = "Not stated" if dx.lower().startswith("not") else dx
        sheet["medicines"] = [{"name": m.get("name"), "dose": (m.get("dosage") if not str(m.get("dosage") or "Not").lower().startswith("not") else "Ask your doctor"),
                               "how_to_take": "Ask your doctor", "duration": "Ask your doctor"}
                              for m in ex.get("medicines", []) if isinstance(m, dict) and str(m.get("status")).lower() != "discussed"]
    sheet.setdefault("diagnosis", "Not stated")
    sheet.setdefault("summary", "")
    sheet.setdefault("follow_up", "Not stated")
    life = sheet.get("lifestyle") if isinstance(sheet.get("lifestyle"), dict) else {}
    sheet["lifestyle"] = {k: [str(x) for x in (life.get(k) or [])] for k in ("diet", "exercise", "monitoring", "other")}
    sheet["medicines"] = [m for m in (sheet.get("medicines") or []) if isinstance(m, dict) and m.get("name")]
    warns = check_sheet(sheet, state["transcript"])
    if failed:
        warns.insert(0, "The language model was unavailable, so this sheet only lists the diagnosis and medicines found by keyword rules. Complete it by hand.")
    return {"sheet": sheet, "sheet_warnings": warns}


# ---------- history and clinic trends ----------

def _record_key(rec: dict):
    """Diabetes type of a saved record: from the stated diagnosis, else from its screening result, else unknown."""
    dx = str(rec.get("diagnosis") or "")
    if dx and not dx.lower().startswith("not"):
        k = type_key(dx)
        if k != "other":
            return k
    top = (rec.get("prediction") or {}).get("possible_types") or []
    return top[0].get("key") if top and isinstance(top[0], dict) else None


def _same_problem(rec: dict, key) -> bool:
    return key in (None, "", "other", "und") or _record_key(rec) in (None, key)


def clinic_trends(similar: list, dx: str, predicted_key: str) -> dict:
    """What this doctor usually prescribes for the same diagnosis or similar patients (counted from saved records)."""
    stated = bool(dx) and not dx.lower().startswith("not")
    key = type_key(dx) if stated else predicted_key
    if key in ("other", "", None) and predicted_key:
        key = predicted_key
    pool = {r["id"]: r for r in similar if _same_problem(r, key)}
    with db() as conn:
        everyone = conn.execute("SELECT COUNT(*) FROM consultations WHERE COALESCE(mode,'lifestyle')='lifestyle'").fetchone()[0]
        if key not in (None, "", "other"):
            for row in conn.execute("SELECT * FROM consultations WHERE COALESCE(mode,'lifestyle')='lifestyle' ORDER BY id DESC LIMIT 500"):
                rec = row_to_dict(row)
                if _record_key(rec) == key:
                    pool.setdefault(rec["id"], rec)

    meds = {}
    for rec in pool.values():
        counted = set()
        for m in rec["medicines"]:
            name = str(m.get("name") or "").strip()
            if not name or str(m.get("status") or "").lower() == "discussed" or name.lower() in counted:
                continue
            counted.add(name.lower())
            entry = meds.setdefault(name.lower(), {"name": name, "count": 0, "doses": Counter()})
            entry["count"] += 1
            dose = str(m.get("dosage") or "").strip()
            if dose and not dose.lower().startswith("not mentioned"):
                entry["doses"][dose] += 1

    total = len(pool)
    ranked = sorted(meds.values(), key=lambda e: -e["count"])[:8]
    approaches = [{"patient_name": r["patient_name"], "date": (r.get("created_at") or "")[:10],
                   "text": r["doctor_approach"][:300]}
                  for r in sorted(pool.values(), key=lambda r: -r["id"]) if (r.get("doctor_approach") or "").strip()][:3]
    label = TYPE_LABELS.get(key, "this problem") if key not in (None, "", "other") else "this problem"
    if everyone == 0:
        note = "Your record book is empty. Save consultations and this doctor's prescribing pattern will build up here."
    elif total == 0:
        note = f"No saved record matches {label} yet ({everyone} consultation{'s' if everyone != 1 else ''} saved for other problems)."
    elif not ranked:
        note = f"{total} similar record{'s' if total != 1 else ''} found, but none lists a prescribed medicine."
    else:
        note = ""
    return {
        "records": total, "key": key, "note": note,
        "medicines": [{"name": e["name"], "count": e["count"], "total": total,
                       "typical_dose": e["doses"].most_common(1)[0][0] if e["doses"] else ""} for e in ranked],
        "approaches": approaches,
    }


def history_node(state: AgentState) -> dict:
    """Earlier visits of this patient, similar patients and prescribing trends from the record book (no LLM)."""
    name = (state.get("patient_name") or "").strip()
    history = []
    if name:
        with db() as conn:
            rows = conn.execute(
                "SELECT * FROM consultations WHERE lower(patient_name)=lower(?) ORDER BY id DESC LIMIT 3", (name,)
            ).fetchall()
        history = [slim(row_to_dict(r)) for r in rows]

    pred = state.get("prediction") or {}
    top = (pred.get("possible_types") or [{}])[0]
    symptoms = ", ".join(s["symptom"] for s in pred.get("symptoms", [])[:6])
    dx = str(state["extracted"].get("diagnosis") or "")
    stated = bool(dx) and not dx.lower().startswith("not")
    key = type_key(dx) if stated and type_key(dx) != "other" else top.get("key", "")

    similar_full = []
    # a conversation with no diabetes picture at all must not be matched to random patients
    has_picture = stated or pred.get("status") in ("preliminary", "complete")
    if has_picture and case_stores["lifestyle"]._collection.count() > 0:
        query = f"{dx}\n{top.get('type', '')}\nSymptoms: {symptoms}\n{state['transcript'][:800]}"
        seen = set()
        with db() as conn:
            for doc in case_stores["lifestyle"].similarity_search(query, k=30):
                rid = doc.metadata.get("record_id")
                if rid in seen:
                    continue
                seen.add(rid)
                row = conn.execute("SELECT * FROM consultations WHERE id=?", (rid,)).fetchone()
                if not row:
                    continue
                rec = row_to_dict(row)
                if name and rec["patient_name"].strip().lower() == name.lower():
                    continue   # this patient's own visits are already shown as history
                if not _same_problem(rec, key):
                    continue   # a different type of diabetes is not a similar patient
                similar_full.append(rec)
                if len(similar_full) == 4:
                    break

    trends = clinic_trends(similar_full, dx, top.get("key", "")) if has_picture else \
        {"records": 0, "key": None, "medicines": [], "approaches": [],
         "note": "Not enough was said to match this conversation to similar patients."}
    return {"patient_history": history, "similar_cases": [slim(r) for r in similar_full], "trends": trends}


# ---------- plan ----------

def plan_node(state: AgentState) -> dict:
    ex = state["extracted"]
    pred = state.get("prediction") or {}
    types = [t["type"] for t in pred.get("possible_types", [])[:2]]
    mode = state.get("mode", DEFAULT_MODE)
    screening_line = (f"POSSIBLE DIABETES TYPES FROM SCREENING: {json.dumps(types)}" if mode == "lifestyle"
                      else f"SETTING: {MODES[mode]['label']}. Audience: {MODES[mode]['audience']}")
    prompt = f"""You plan evidence retrieval for a clinical decision-support assistant.
From this consultation, write 2 or 3 short search questions to look up BEFORE any advice is given.

Tools:
- guidelines: clinical guidelines and reference documents (standard medicines, targets, complications, lifestyle, follow-up tests)
- clinic_records: this clinic's earlier consultations (how similar patients were handled)

Return ONLY a JSON object:
{{"questions": [{{"question": "...", "tool": "guidelines" or "clinic_records"}}]}}

Rules: at least one question for guidelines, at most one for clinic_records. Each question must stand alone,
name the condition, and be under 20 words. Cover treatment, lifestyle and the tests that confirm the diagnosis.

DOCTOR'S EXTRACTED DETAILS: {json.dumps(ex)}
{screening_line}
TRANSCRIPT (start): {state["transcript"][:1500]}"""
    try:
        data = ask(prompt, as_json=True)
    except RuntimeError:
        data = {}
    raw = data.get("questions") if isinstance(data, dict) else None

    questions = []
    for item in raw or []:
        if isinstance(item, dict) and str(item.get("question") or "").strip():
            tool = item.get("tool") if item.get("tool") in TOOLS else "guidelines"
            questions.append({"question": str(item["question"]).strip(), "tool": tool})
    questions = questions[:3]

    if not any(q["tool"] == "guidelines" for q in questions):
        dx = str(ex.get("diagnosis") or "")
        dx = (types[0] if types else "diabetes") if not dx or dx.lower().startswith("not") else dx
        questions = [{"question": f"Standard first-line treatment, monitoring and lifestyle advice for {dx}",
                      "tool": "guidelines"}] + questions[:2]
    for i, q in enumerate(questions, 1):
        q["id"] = f"Q{i}"
    return {"questions": questions}


# ---------- retrieve ----------

def run_tool(tool: str, query: str, mode: str = DEFAULT_MODE):
    """Search one data source. Returns a list of hits, or None when that source is empty."""
    if tool == "clinic_records":
        store = case_stores[mode]
        if store._collection.count() == 0:
            return None
        hits = []
        with db() as conn:
            for doc in store.similarity_search(query, k=RETRIEVAL_K):
                row = conn.execute("SELECT patient_name, created_at FROM consultations WHERE id=?",
                                   (doc.metadata.get("record_id"),)).fetchone()
                if row:
                    hits.append({"source": f"Clinic record: {row['patient_name']}",
                                 "page_label": row["created_at"][:10], "text": doc.page_content})
        return hits
    return kb_search(mode, query, RETRIEVAL_K)


def grade_passages(searched: list, by_id: dict) -> set:
    """One LLM call grades every new passage for every question. Returns the ids judged relevant."""
    blocks = []
    for q in searched:
        passages = "\n".join(f"[{i}] {by_id[i]['text']}" for i in q["ids"])
        blocks.append(f"QUESTION {q['id']}: {q['query']}\nPASSAGES:\n{passages}")
    joined = "\n\n".join(blocks)
    prompt = f"""You grade search results for a clinical assistant.
For each question, decide which passages contain facts that directly help answer it.
Be strict: skip passages that only share a keyword with the question.

Return ONLY a JSON object: {{"relevant_ids": ["E1", "E3"]}}   (use [] if none are relevant)

{joined}"""
    data = ask(prompt, as_json=True)
    ids = data.get("relevant_ids") if isinstance(data, dict) else []
    return {str(i).strip() for i in (ids or [])}


def rewrite_queries(pending: list) -> list:
    """Ask for a better search query for each weak question. Returns the questions that were rewritten."""
    lines = "\n".join(f"{q['id']}: {q['query']}" for q in pending)
    prompt = f"""These search queries returned no relevant passages from a clinical knowledge base.
Rewrite each one with different, more specific medical wording (synonyms, drug classes, guideline terms).

Return ONLY a JSON object: {{"queries": {{"Q1": "new query", "Q2": "new query"}}}}

{lines}"""
    try:
        data = ask(prompt, as_json=True)
    except RuntimeError:
        data = {}
    new = data.get("queries") if isinstance(data, dict) and isinstance(data.get("queries"), dict) else {}
    kept = []
    for q in pending:
        text = str(new.get(q["id"]) or "").strip()
        if text and text != q["query"]:
            q["query"] = text
            q["rewrites"] += 1
            kept.append(q)
        else:
            q["status"] = "insufficient"
    return kept


def retrieve_node(state: AgentState) -> dict:
    questions = [dict(q, status="pending", retrieved=0, relevant=0, rewrites=0, query=q["question"], ids=[])
                 for q in state["questions"]]
    evidence, registry, steps = [], {}, []
    warnings = list(state.get("warnings", []))
    by_id = {}
    pending = list(questions)
    rounds = 0

    while pending and rounds < MAX_ROUNDS:
        rounds += 1
        searched = []
        for q in pending:
            hits = run_tool(q["tool"], q["query"], state.get("mode", DEFAULT_MODE))
            if hits is None:
                q["status"] = "no_source" if q["tool"] == "guidelines" else "none_found"
                steps.append(f"{q['id']}: {q['tool']} has no data yet")
                continue
            ids = []
            for h in hits:
                key = (h["source"], h["page_label"], hash(h["text"]))
                if key not in registry:
                    entry = {"id": f"E{len(evidence) + 1}", "source": h["source"], "page_label": h["page_label"],
                             "snippet": h["text"][:300], "text": h["text"][:700], "relevant": False}
                    registry[key] = entry
                    evidence.append(entry)
                    by_id[entry["id"]] = entry
                if registry[key]["id"] not in ids:
                    ids.append(registry[key]["id"])
            q["ids"], q["retrieved"] = ids, len(ids)
            steps.append(f"{q['id']}: searched {q['tool']} for \"{q['query']}\" ({len(ids)} passages)")
            if ids:
                searched.append(q)
            else:
                q["status"] = "insufficient"
        if not searched:
            pending = []
            break

        try:
            relevant_ids = grade_passages(searched, by_id)
        except RuntimeError as error:
            relevant_ids = set()
            warnings.append("Relevance grading failed, so retrieved passages were treated as unverified.")
            print(f"[agent] grading failed: {error}")
        for e in evidence:
            if e["id"] in relevant_ids:
                e["relevant"] = True

        pending = []
        for q in searched:
            q["relevant"] = sum(1 for i in q["ids"] if by_id[i]["relevant"])
            if q["relevant"]:
                q["status"] = "sufficient"
            elif q["rewrites"] < 1 and rounds < MAX_ROUNDS:
                pending.append(q)
            else:
                q["status"] = "insufficient"
        if pending:
            steps.append(f"Weak evidence for {', '.join(q['id'] for q in pending)}; rewriting the queries")
            pending = rewrite_queries(pending)

    for q in questions:
        if q["status"] == "pending":
            q["status"] = "insufficient"
    return {"questions": questions, "evidence": evidence, "rounds": rounds, "steps": steps, "warnings": warnings}


# ---------- advise ----------

ADVICE_KEYS = ("standard_medicines", "possible_medicines", "diet", "activity", "lifestyle",
               "follow_up", "diagnosis_considerations")
MEDICINE_KEYS = ("standard_medicines", "possible_medicines")


# ---------- built-in reference (used only when the model or the documents give nothing for a section) ----------
# General, guideline-level statements (ADA / WHO style). They are shown as "unverified": the doctor decides.
# Have your clinical lead review this table, as with app/triage.py.

_M = lambda name, cls, why: {"name": name, "drug_class": cls, "why": why}   # noqa: E731

STANDARD_CARE = {
    "t2": {
        "medicines": [_M("Metformin", "Biguanide", "Usual first-line medicine unless kidney function is poor or it is not tolerated."),
                      _M("SGLT2 inhibitor (e.g. empagliflozin, dapagliflozin)", "SGLT2 inhibitor", "Add when there is heart failure, chronic kidney disease or established heart disease."),
                      _M("GLP-1 receptor agonist (e.g. semaglutide, liraglutide)", "GLP-1 receptor agonist", "Add when weight loss or heart protection is a priority."),
                      _M("Sulfonylurea (e.g. glimepiride, gliclazide)", "Sulfonylurea", "Low-cost add-on; watch for low blood sugar."),
                      _M("DPP-4 inhibitor (e.g. sitagliptin, linagliptin)", "DPP-4 inhibitor", "Well-tolerated add-on with little risk of low blood sugar.")],
        "possible": [_M("Basal insulin", "Insulin", "Needed when HbA1c is very high, there are weight loss or ketone symptoms, or tablets are not enough."),
                     _M("Pioglitazone", "Thiazolidinedione", "Option when insulin resistance is prominent and there is no heart failure.")],
        "considerations": ["Check kidney function (creatinine/eGFR, urine albumin) before metformin or an SGLT2 inhibitor.",
                           "Screen for complications now: eyes (retinopathy), feet (neuropathy), kidneys, blood pressure and lipids.",
                           "If the patient is lean, young or losing weight fast, consider autoimmune diabetes (LADA) and check antibodies and C-peptide."],
        "follow_up": ["Review HbA1c in about 3 months and set an individual target (often below 7%).",
                      "Review the kidney, lipid and eye results and the patient's home glucose readings.",
                      "Recheck weight, blood pressure and foot examination at each visit."],
    },
    "t1": {
        "medicines": [_M("Basal insulin (e.g. glargine, detemir, degludec)", "Long-acting insulin", "Background insulin; always required in type 1 diabetes."),
                      _M("Rapid-acting insulin (e.g. lispro, aspart)", "Rapid-acting insulin", "Mealtime and correction doses."),
                      _M("Glucagon rescue kit", "Glucagon", "For severe low blood sugar.")],
        "possible": [_M("Human regular or NPH insulin", "Insulin", "Lower-cost alternative regimen where analogues are not available.")],
        "considerations": ["Check blood or urine ketones and a venous gas now; ketoacidosis needs same-day hospital care.",
                           "Confirm the type with islet autoantibodies (GAD65, IA-2, ZnT8) and C-peptide.",
                           "Screen for other autoimmune disease: thyroid antibodies and coeliac serology."],
        "follow_up": ["Teach insulin technique, carbohydrate counting, low-sugar recognition and sick-day rules.",
                      "Arrange home glucose monitoring (or a continuous glucose monitor) and review the readings.",
                      "Review HbA1c in about 3 months; start eye, kidney and foot screening after the first years."],
    },
    "pre": {
        "medicines": [],
        "note": "No medicine is first-line for prediabetes: lifestyle change comes first. Metformin is an option for high-risk patients.",
        "possible": [_M("Metformin", "Biguanide", "Can be considered in high-risk people (BMI 35 or more, age under 60, past gestational diabetes) if lifestyle change is not enough.")],
        "considerations": ["Confirm with a 2-hour OGTT or a repeat HbA1c / fasting glucose, since these tests can disagree.",
                           "Check blood pressure and lipids: cardiovascular risk is raised.",
                           "Ask about past gestational diabetes, PCOS and family history."],
        "follow_up": ["Repeat HbA1c or fasting glucose in 3 to 6 months, then at least yearly.",
                      "Set a weight target (5 to 7 percent loss if overweight) and review progress."],
    },
    "gdm": {
        "medicines": [_M("Insulin", "Insulin", "Preferred medicine when diet and activity do not reach glucose targets in pregnancy.")],
        "possible": [_M("Metformin", "Biguanide", "Alternative or add-on in some guidelines; crosses the placenta, so discuss with the obstetrician.")],
        "considerations": ["Confirm with a 75 g OGTT (24 to 28 weeks; earlier if risk factors).",
                           "Plan fetal growth scans and delivery timing with the obstetrician.",
                           "Check for symptoms of hypoglycemia if medicines are started."],
        "follow_up": ["Review fasting and after-meal home glucose readings every 1 to 2 weeks.",
                      "Arrange a 75 g OGTT 4 to 12 weeks after delivery, then screening at least every 1 to 3 years."],
    },
    "lada": {
        "medicines": [_M("Insulin", "Insulin", "Started early because beta-cell failure is expected.")],
        "possible": [_M("Metformin", "Biguanide", "Possible add-on in a patient with insulin resistance; not a substitute for insulin when C-peptide is low."),
                     _M("DPP-4 inhibitor", "DPP-4 inhibitor", "Sometimes used early; sulfonylureas are generally avoided because they may speed beta-cell loss.")],
        "considerations": ["Confirm with GAD65 antibodies and C-peptide.",
                           "Consider that the patient was labelled type 2 but is not responding to tablets.",
                           "Screen for other autoimmune disease (thyroid, coeliac)."],
        "follow_up": ["Review antibody and C-peptide results and decide the insulin plan.",
                      "Review HbA1c in about 3 months and teach low-sugar recognition."],
    },
    "mody": {
        "medicines": [_M("Low-dose sulfonylurea (e.g. gliclazide)", "Sulfonylurea", "Often effective in HNF1A / HNF4A MODY; the right medicine depends on the gene.")],
        "possible": [_M("No medicine", "-", "GCK-MODY usually needs no treatment; confirm the gene before starting anything."),
                     _M("Insulin", "Insulin", "If the sulfonylurea fails or in pregnancy, depending on the subtype.")],
        "considerations": ["Genetic testing for the MODY genes is the confirming test; ask for family members' results.",
                           "A strong multi-generation family history and young onset support the suspicion."],
        "follow_up": ["Send the genetic test and review the result before choosing treatment.",
                      "Offer family counselling and testing for relatives."],
    },
    "sec": {
        "medicines": [_M("Insulin", "Insulin", "Often needed for steroid-induced or severe secondary hyperglycemia.")],
        "possible": [_M("Metformin", "Biguanide", "Possible once kidney function is checked and the cause is addressed.")],
        "considerations": ["Review every medicine that can raise blood sugar (steroids, some antipsychotics, thiazides).",
                           "Look for pancreatic or endocrine causes (pancreatitis, Cushing syndrome, acromegaly, hemochromatosis) as clinically indicated."],
        "follow_up": ["Reduce or change the causing medicine where it is safe, and recheck glucose after the change.",
                      "Repeat HbA1c or fasting glucose after the cause is treated to see whether diabetes persists."],
    },
    "und": {
        "note": "The medicine depends on the diabetes type. Confirm the diagnosis and type first.",
        "medicines": [],
        "possible": [_M("Insulin", "Insulin", "Only if there are ketones, weight loss or very high glucose."),
                     _M("Metformin", "Biguanide", "Reasonable once type 2 is likely and kidney function is checked.")],
        "considerations": ["Confirm diabetes with HbA1c, fasting glucose or an OGTT, and repeat it if there are no clear symptoms.",
                           "Decide the type with antibodies and C-peptide when the picture is not typical of type 2."],
        "follow_up": ["Review the confirming tests and choose treatment by type at the next visit."],
    },
    "insufficient": {
        "note": "Not enough was said to suggest any medicine.",
        "medicines": [], "possible": [],
        "considerations": ["There is not enough in this conversation to suggest a diabetes type or a medicine."],
        "follow_up": ["Ask about thirst, passing urine, weight change, family history and earlier sugar reports. Test HbA1c or fasting glucose if risk factors exist."],
    },
}

CARE_LIFESTYLE = {
    "diet": ["Cut sugary drinks and sweets; build meals around vegetables, pulses, whole grains and lean protein.",
             "Keep meal times and portions steady, with carbohydrate spread across the day."],
    "activity": ["Aim for about 150 minutes a week of brisk walking or similar, in sessions of at least 10 minutes."],
    "lifestyle": ["Check the feet daily and report any wound that does not heal.", "Stop smoking and keep alcohol low.", "Sleep 7 to 8 hours and manage stress."],
}
CARE_LIFESTYLE_BY_TYPE = {
    "t1": {"diet": ["Match mealtime insulin to carbohydrate (carbohydrate counting) with a diabetes dietitian."],
           "activity": ["Check glucose before, during and after exercise and carry fast-acting sugar."],
           "lifestyle": ["Learn sick-day rules and when to check ketones.", "Carry fast-acting glucose and learn to recognise low blood sugar."]},
    "gdm": {"diet": ["Small regular meals, limit refined carbohydrate, with a dietitian review."],
            "activity": ["A 10 to 15 minute walk after meals unless the obstetrician advises against it."],
            "lifestyle": ["Check home glucose as advised (fasting and after meals)."]},
}

BUILT_IN_NOTE = "Built-in reference, not from your documents. Verify against your guideline."


def care_key(dx: str, predicted: dict) -> str:
    """Which row of STANDARD_CARE fits this patient: stated diagnosis first, then the screening result."""
    stated = bool(dx) and not dx.lower().startswith("not")
    k = type_key(dx) if stated else "other"
    if k in STANDARD_CARE and k not in ("und",):
        return k
    top = (predicted.get("possible_types") or [{}])[0].get("key")
    if top in STANDARD_CARE:
        return top
    if not stated and predicted.get("status") == "insufficient":
        return "insufficient"
    return "und"


def built_in_advice(key: str) -> dict:
    care = STANDARD_CARE[key]
    life = {k: CARE_LIFESTYLE_BY_TYPE.get(key, {}).get(k) or ([] if key == "insufficient" else CARE_LIFESTYLE[k]) for k in CARE_LIFESTYLE}
    texts = lambda lines: [{"text": t} for t in lines]   # noqa: E731
    return {"standard_medicines": care["medicines"], "possible_medicines": care["possible"],
            "diet": texts(life["diet"]), "activity": texts(life["activity"]), "lifestyle": texts(life["lifestyle"]),
            "follow_up": texts(care["follow_up"]), "diagnosis_considerations": texts(care["considerations"])}


def advice_node(state: AgentState) -> dict:
    ex = state["extracted"]
    pred = state.get("prediction") or {}
    types = [{"type": t["type"], "likelihood": t["likelihood"]} for t in pred.get("possible_types", [])[:3]]
    tests = [t["test"] for t in pred.get("tests", [])]
    trends = [f"{m['name']} ({m['count']}/{m['total']} similar records)" for m in (state.get("trends") or {}).get("medicines", [])]
    relevant = [e for e in state.get("evidence", []) if e["relevant"]]
    evidence_text = "\n\n".join(f"[{e['id']}] ({e['source']} {e['page_label']})\n{e['text']}" for e in relevant) or "None."

    prompt = f"""You are a clinical decision-support assistant for a qualified doctor.

Return ONLY a JSON object:

{{
  "standard_medicines": [{{"name": "...", "drug_class": "...", "why": "one line: its role in standard care", "evidence": ["E1"]}}],
  "possible_medicines": [{{"name": "...", "drug_class": "...", "why": "when it might be needed, e.g. once tests confirm the type", "evidence": []}}],
  "diet": [{{"text": "specific food and meal advice", "evidence": []}}],
  "activity": [{{"text": "specific physical activity advice", "evidence": []}}],
  "lifestyle": [{{"text": "other lifestyle change: weight, sleep, smoking, alcohol, stress, foot care, home glucose checks", "evidence": []}}],
  "follow_up": [{{"text": "what to check or do at the NEXT meeting: tests to review, targets, when to return", "evidence": []}}],
  "diagnosis_considerations": [{{"text": "differential, complication or missing assessment worth checking", "evidence": ["E2"]}}]
}}

Rules:
- "evidence" lists the ids of the EVIDENCE passages below that support the item. Use only ids that appear below.
- If no passage supports an item, give it an empty evidence list. Never invent an id or a citation.
- standard_medicines: the medicines physicians commonly use for this diagnosis (first-line and usual add-ons). Max 5. No patient-specific doses.
- possible_medicines: medicines that might be needed for this patient depending on test results, kidney function or response. Max 5.
- Use the diagnosis the doctor stated. If none was stated, work from the most likely screening result and say it is unconfirmed.
- diet, activity, lifestyle and follow_up must fit this patient's own routine and symptoms from the transcript. Max 6 items each.
- Everything is a suggestion for the doctor to verify; never claim a decision has been made.

DOCTOR'S EXTRACTED DETAILS: {json.dumps(ex)}
SCREENING RESULT: {json.dumps(types)}
TESTS SUGGESTED TO CONFIRM: {json.dumps(tests)}
WHAT THIS CLINIC USUALLY PRESCRIBES (context only, not a citation): {json.dumps(trends)}
EVIDENCE:
{evidence_text}
TRANSCRIPT:
{state['transcript']}"""
    warnings = list(state.get("warnings", []))
    try:
        raw = ask(prompt, as_json=True)
    except RuntimeError as error:
        print(f"[agent] advice failed, using the built-in reference: {error}")
        raw = {}
        warnings.append("The language model was unavailable, so medicines and the plan come from the built-in reference. Verify them.")
    raw = raw if isinstance(raw, dict) else {}

    valid = {e["id"] for e in relevant}
    advice = {}
    for key in ADVICE_KEYS:
        items = []
        for it in raw.get(key) or []:
            if isinstance(it, str):
                it = {"text": it, "evidence": []}
            if not isinstance(it, dict):
                continue
            if key in MEDICINE_KEYS:
                if not str(it.get("name") or "").strip():
                    continue
            elif not str(it.get("text") or "").strip():
                continue
            cited = it.get("evidence") if isinstance(it.get("evidence"), list) else []
            it["evidence"] = [str(i) for i in cited if str(i) in valid]
            it["supported"] = bool(it["evidence"])
            it["note"] = "" if it["supported"] else "No retrieved source supports this line."
            items.append(it)
        advice[key] = items[:5] if key in MEDICINE_KEYS else items[:6]

    # every section must answer: where the model gave nothing, fill it from the built-in reference
    fallback = built_in_advice(care_key(str(ex.get("diagnosis") or ""), pred))
    filled = []
    for key in ADVICE_KEYS:
        if not advice[key] and fallback.get(key):
            advice[key] = [dict(it, evidence=[], supported=False, note=BUILT_IN_NOTE, source="built-in") for it in fallback[key]]
            filled.append(key)
    if not advice["standard_medicines"]:
        advice["medicine_note"] = STANDARD_CARE[care_key(str(ex.get("diagnosis") or ""), pred)].get("note", "")
    if filled and not any(w.startswith("The language model was unavailable") for w in warnings):
        warnings.append("The model gave nothing for some sections, so they show the built-in reference (marked unverified).")
    return {"advice": advice, "warnings": warnings}


# ---------- finalize ----------

def finalize_node(state: AgentState) -> dict:
    questions, evidence, advice = state["questions"], state["evidence"], state["advice"]
    warnings = list(state.get("warnings", []))

    answerable = [q for q in questions if q["status"] in ("sufficient", "insufficient")]
    relevant_total = sum(1 for e in evidence if e["relevant"])
    if not answerable or relevant_total == 0:
        confidence = "insufficient"
    elif all(q["status"] == "sufficient" for q in answerable):
        confidence = "sufficient"
    else:
        confidence = "partial"

    if any(q["status"] == "no_source" for q in questions):
        warnings.append("No guideline files are loaded for this mode, so suggestions cannot be checked against guidelines. "
                        "Put PDFs in knowledge/ and run the ingest.")
    for q in questions:
        if q["status"] == "insufficient":
            warnings.append(f"{q['id']}: no relevant evidence found for \"{q['question']}\".")
    if str(state["extracted"].get("diagnosis", "")).lower().startswith("not"):
        warnings.append("The doctor did not state a diagnosis, so suggestions follow the screening result and are unconfirmed.")

    all_items = [it for key in ADVICE_KEYS for it in advice.get(key, [])]
    metrics = {
        "retrieval_precision": round(relevant_total / len(evidence), 2) if evidence else None,
        "grounded_items": sum(1 for it in all_items if it["supported"]),
        "total_items": len(all_items),
    }
    agent = {
        "rounds": state.get("rounds", 0),
        "questions": [{k: v for k, v in q.items() if k != "ids"} for q in questions],
        "metrics": metrics,
        "steps": state.get("steps", []),
    }
    return {"confidence": confidence, "warnings": warnings, "agent": agent}


# ---------- PDF ----------

def build_sheet_pdf(patient: str, doctor: str, sheet: dict) -> bytes:
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    teal = colors.HexColor("#0d7c80")
    base = getSampleStyleSheet()["BodyText"]
    body = ParagraphStyle("b", parent=base, fontSize=11, leading=15)
    cell = ParagraphStyle("c", parent=body, fontSize=10, leading=13)
    head = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=13, textColor=teal, spaceBefore=12, spaceAfter=4)
    title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=22, leading=26)
    P = lambda t, st=body: Paragraph(escape(str(t or "")).replace("\n", "<br/>"), st)

    story = [P("Your Care Sheet", title), Spacer(1, 4)]
    meta = [[Paragraph(f"<b>Patient:</b> {escape(patient)}", cell), Paragraph(f"<b>Date:</b> {datetime.now():%d %b %Y}", cell)]]
    if doctor:
        meta.append([Paragraph(f"<b>Doctor:</b> {escape(doctor)}", cell), ""])
    t = Table(meta, colWidths=[95 * mm, 75 * mm])
    t.setStyle(TableStyle([("LINEBELOW", (0, -1), (-1, -1), 1, teal), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [t, Paragraph("Your diagnosis", head), P(sheet.get("diagnosis") or "Not stated")]
    if sheet.get("summary"):
        story += [Paragraph("In simple words", head), P(sheet["summary"])]

    story.append(Paragraph("Your medicines", head))
    meds = sheet.get("medicines") or []
    if meds:
        rows = [[P(h, ParagraphStyle("th", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)) for h in
                 ("Medicine", "Dose", "How and when to take", "For how long")]]
        for m in meds:
            rows.append([P(m.get("name"), cell), P(m.get("dose"), cell), P(m.get("how_to_take"), cell), P(m.get("duration"), cell)])
        mt = Table(rows, colWidths=[42 * mm, 28 * mm, 62 * mm, 38 * mm], repeatRows=1)
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), teal),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8c7ca")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f8f8")]),
        ]))
        story.append(mt)
    else:
        story.append(P("No medicines were prescribed in this consultation."))

    plan = sheet.get("plan") or []
    if plan:      # emergency, critical care and childbirth sheets use their own section titles
        story.append(Paragraph(escape(str(sheet.get("plan_title") or "What to do")), head))
        for sec in plan:
            if sec.get("items"):
                story.append(Paragraph(f"<b>{escape(str(sec.get('title') or ''))}</b>", body))
                story += [P("\u2022  " + item) for item in sec["items"]]
                story.append(Spacer(1, 4))
    else:
        labels = {"diet": "Food and drink", "exercise": "Activity", "monitoring": "Checks to do at home", "other": "Other advice"}
        life = sheet.get("lifestyle") or {}
        story.append(Paragraph("Lifestyle advice from your doctor", head))
        if any(life.get(k) for k in labels):
            for key, label in labels.items():
                if life.get(key):
                    story.append(Paragraph(f"<b>{label}</b>", body))
                    story += [P("\u2022  " + item) for item in life[key]]
                    story.append(Spacer(1, 4))
        else:
            story.append(P("No lifestyle advice was recorded."))

    story += [Paragraph("Next visit", head), P(sheet.get("follow_up") or "Not stated")]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5d7077"))
        canvas.drawString(20 * mm, 12 * mm, "Prepared from your consultation and reviewed by your doctor. Follow your doctor's instructions.")
        canvas.drawString(20 * mm, 8 * mm, "If you feel worse or have a problem with a medicine, contact your doctor or clinic.")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=22 * mm,
                      title="Care Sheet").build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def pdf_response(patient: str, doctor: str, sheet: dict) -> Response:
    safe = "".join(c if c.isalnum() else "-" for c in patient) or "patient"
    return Response(
        build_sheet_pdf(patient, doctor, sheet),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="care-sheet-{safe}.pdf"'},
    )


graph = StateGraph(AgentState)
graph.add_node("transcribe", transcribe_node)
graph.add_node("extract", extract_node)
graph.add_node("predict", predict_node)
graph.add_node("sheet", sheet_node)
graph.add_node("history", history_node)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("advise", advice_node)
graph.add_node("finalize", finalize_node)
graph.add_edge(START, "transcribe")
graph.add_edge("transcribe", "extract")
graph.add_edge("extract", "predict")
graph.add_edge("predict", "sheet")
graph.add_edge("sheet", "history")
graph.add_edge("history", "plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "advise")
graph.add_edge("advise", "finalize")
graph.add_edge("finalize", END)
agent_app = graph.compile()

# ============================================================
# 7. Saving a consultation to the vector database
# ============================================================

convo_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def index_record(record_id: int, patient: str, diagnosis: str, approach: str, medicines: list,
                 prediction: dict, transcript: str, mode: str = DEFAULT_MODE):
    """Embed one summary document plus the full conversation (in chunks) so later searches can find it."""
    meds = ", ".join(f"{m.get('name')} {m.get('dosage', '')}".strip() for m in medicines if isinstance(m, dict))
    types = ", ".join(t.get("type", "") for t in (prediction.get("possible_types") or [])[:3])
    symptoms = ", ".join(s.get("symptom", "") for s in (prediction.get("symptoms") or []))
    summary = (f"Diagnosis: {diagnosis}\nPossible types: {types}\nSymptoms: {symptoms}\n"
               f"Medicines given: {meds}\nApproach: {approach}")
    docs = [Document(page_content=summary, metadata={"record_id": record_id, "patient": patient, "kind": "summary"})]
    for i, chunk in enumerate(convo_splitter.split_text(transcript or "")):
        docs.append(Document(page_content=f"Diagnosis: {diagnosis}\nConversation: {chunk}",
                             metadata={"record_id": record_id, "patient": patient, "kind": "conversation", "chunk": i}))
    case_stores.get(mode, case_stores[DEFAULT_MODE]).add_documents(docs)


# ============================================================
# 8. Childbirth: same agentic retrieval, with its own advice step
#
#   transcribe -> extract -> plan -> retrieve -> advise -> finalize
# ============================================================


def mode_advise_node(state: AgentState) -> dict:
    return {"guidance": agentic_guidance(state["mode"], state["transcript"], state["extracted"], state.get("evidence", []))}


def mode_finalize_node(state: AgentState) -> dict:
    questions, evidence, guidance = state["questions"], state.get("evidence", []), state["guidance"]
    warnings = list(state.get("warnings", []))
    answerable = [q for q in questions if q["status"] in ("sufficient", "insufficient")]
    relevant_total = sum(1 for e in evidence if e["relevant"])
    if not answerable or relevant_total == 0:
        confidence = "insufficient"
    elif all(q["status"] == "sufficient" for q in answerable):
        confidence = "sufficient"
    else:
        confidence = "partial"
    if any(q["status"] == "no_source" for q in questions):
        warnings.append("No documents are loaded for this mode, so nothing can be checked against a source. "
                        "Add files to the matching knowledge folder and run the ingest.")
    for q in questions:
        if q["status"] == "insufficient":
            warnings.append(f"{q['id']}: no relevant evidence found for \"{q['question']}\".")
    items = [i for k in ("steps", "avoid", "escalate") for i in guidance.get(k, [])]
    metrics = {"retrieval_precision": round(relevant_total / len(evidence), 2) if evidence else None,
               "grounded_items": sum(1 for i in items if i["supported"]), "total_items": len(items)}
    agent = {"rounds": state.get("rounds", 0), "metrics": metrics, "steps": state.get("steps", []),
             "questions": [{k: v for k, v in q.items() if k != "ids"} for q in questions]}
    return {"confidence": confidence, "warnings": warnings, "agent": agent}


mode_graph = StateGraph(AgentState)
for _name, _fn in (("transcribe", transcribe_node), ("extract", extract_node), ("plan", plan_node),
                   ("retrieve", retrieve_node), ("advise", mode_advise_node), ("finalize", mode_finalize_node)):
    mode_graph.add_node(_name, _fn)
mode_graph.add_edge(START, "transcribe")
for _a, _b in (("transcribe", "extract"), ("extract", "plan"), ("plan", "retrieve"),
               ("retrieve", "advise"), ("advise", "finalize")):
    mode_graph.add_edge(_a, _b)
mode_graph.add_edge("finalize", END)
mode_app = mode_graph.compile()


# ============================================================
# 9. FastAPI
# ============================================================



def _rebuild_case_index():
    """Hosted disks are wiped on restart, so the vector index of past consultations is rebuilt from the database."""
    with db() as conn:
        rows = [row_to_dict(r) for r in conn.execute("SELECT * FROM consultations ORDER BY id")]
    for rec in rows:
        mode = rec.get("mode") or DEFAULT_MODE
        try:
            case_stores[mode].delete(where={"record_id": rec["id"]})
            index_record(rec["id"], rec["patient_name"], rec["diagnosis"] or "", rec["doctor_approach"] or "",
                         rec["medicines"], rec["prediction"], rec["transcript"] or "", mode)
        except Exception as error:
            print(f"[kb] could not index record {rec['id']}: {error}")
    print(f"[kb] rebuilt the case index from {len(rows)} saved consultations")


def _startup_indexing():
    """Runs in the background so uvicorn opens its port at once (Render fails deploys that bind too late)."""
    try:
        if os.getenv("AUTO_INGEST", "true").lower() == "true":
            ingest_all()   # embeds only new or changed files
        if USE_POSTGRES and os.getenv("REBUILD_CASE_INDEX_ON_START", "true").lower() == "true":
            _rebuild_case_index()
    except Exception as error:
        print(f"[startup] indexing failed: {type(error).__name__}: {str(error)[:300]}")


threading.Thread(target=_startup_indexing, daemon=True, name="startup-indexing").start()

app = FastAPI(title="Consultation Copilot", version="5.0.0")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    """Optional login for the whole app (set APP_PASSWORD). /health stays open for Docker health checks."""
    if not APP_PASSWORD or request.url.path == "/health":
        return await call_next(request)
    ok = False
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
            ok = secrets.compare_digest(user.encode(), APP_USER.encode()) and \
                secrets.compare_digest(pw.encode(), APP_PASSWORD.encode())
        except Exception:
            ok = False
    if not ok:
        return Response("Login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Consultation Copilot"'})
    return await call_next(request)


def _tailwind_tag() -> str:
    """Built CSS when it exists (Docker build / npm run build:css); otherwise the Tailwind CDN so the page still renders."""
    css = STATIC_DIR / "tailwind.css"
    if css.exists():
        return f'<link rel="stylesheet" href="/static/tailwind.css?v={int(css.stat().st_mtime)}">'
    try:
        config = (FRONTEND_DIR / "tailwind.config.js").read_text(encoding="utf-8")
        config = re.sub(r"module\.exports\s*=", "tailwind.config =", config)
        source = (FRONTEND_DIR / "src" / "input.css").read_text(encoding="utf-8")
    except OSError:
        return ""
    return (f'<script src="https://cdn.tailwindcss.com/3.4.17"></script><script>{config}</script>'
            f'<style type="text/tailwindcss">{source}</style>')


@app.get("/", response_class=HTMLResponse)
async def frontend():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("<!--TAILWIND-->", _tailwind_tag()), headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health():
    with db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM consultations").fetchone()[0]
    return {"status": "running", "models": [name for name, _ in LLMS], "saved_consultations": n,
            "knowledge": kb_status()}


@app.get("/modes")
async def modes():
    return {"emergency_number": EMERGENCY_NUMBER, "ambulance_number": AMBULANCE_NUMBER, "default": DEFAULT_MODE,
            "modes": [{"id": k, **{x: v[x] for x in ("label", "short", "pipeline", "live_voice")}, "chunks": mode_chunks(k)}
                      for k, v in MODES.items()]}


@app.get("/protocols")
async def protocols(mode: str = "emergency"):
    """Built-in first-aid cards. The page loads these once, so the tiles work instantly."""
    return {"emergency_number": EMERGENCY_NUMBER, "ambulance_number": AMBULANCE_NUMBER,
            "protocols": all_protocols(clinician=(mode == "critical"))}


@app.post("/triage")
async def triage_route(text: str = Form(""), mode: str = Form("emergency")):
    """Instant, no model: which built-in protocol matches this text?"""
    return triage(text, clinician=(mode == "critical"))


@app.get("/kb/status")
async def kb_state():
    return kb_status()


@app.post("/kb/reindex")
async def kb_reindex(rebuild: bool = Form(False), only: str = Form("")):
    """After adding files to knowledge/: embed the new ones into the vector databases."""
    reports = await run_in_threadpool(ingest_all, rebuild, only or None)
    return {"reports": reports, "knowledge": kb_status()}


def _public_evidence(evidence: list) -> list:
    return [{k: v for k, v in e.items() if k != "text"} for e in evidence]


async def _run_fast(mode: str, start: dict) -> dict:
    """Emergency and critical care: show the built-in steps at once, ask the model once, never wait forever."""
    transcript = start.get("transcript") or await run_in_threadpool(transcribe, start["audio_path"])
    if not transcript:
        raise ValueError("No speech was detected in the recording.")
    tri = triage(transcript, clinician=(mode == "critical"))
    guidance, evidence, warnings, status = None, [], [], "ok"
    try:
        guidance, evidence, warnings = await asyncio.wait_for(
            run_in_threadpool(fast_guidance, mode, transcript, tri), FAST_TIMEOUT)
    except asyncio.TimeoutError:
        status = "timeout"
        warnings = [f"The AI model did not answer within {FAST_TIMEOUT:.0f} seconds. The built-in steps are still valid: follow them now."]
    except Exception as error:
        status = "failed"
        print(f"[fast] model unavailable: {type(error).__name__}: {str(error)[:200]}")
        warnings = ["The AI model is unavailable right now. The built-in steps are still valid: follow them now."]
    headline = (guidance or {}).get("headline") or (tri["protocols"][0]["headline"] if tri["protocols"] else "")
    return {
        "mode": mode, "transcript": transcript,
        "extracted": {"dialogue": split_dialogue(transcript), "diagnosis": headline or "Not mentioned", "medicines": [],
                      "vitals": "Not mentioned", "doctor_approach": (guidance or {}).get("summary", "")},
        "triage": tri, "guidance": guidance, "evidence": _public_evidence(evidence),
        "confidence": confidence_of(guidance, evidence) if guidance else "insufficient",
        "warnings": warnings, "llm_status": status,
        "agent": {"rounds": 1, "questions": [], "metrics": {}, "steps": [
            f"Built-in triage matched: {', '.join(p['title'] for p in tri['protocols']) or 'nothing'}",
            f"Searched {len(evidence)} passages, one model call ({status})"]},
    }


@app.post("/analyze")
async def analyze(
    audio_file: Optional[UploadFile] = File(None),
    patient_name: str = Form(""),
    transcript_text: str = Form(""),
    mode: str = Form(DEFAULT_MODE),
):
    """Analyse a recorded WAV conversation, or typed/pasted text, for the chosen mode."""
    if mode not in MODES:
        raise HTTPException(400, f"Unknown mode. Use one of: {', '.join(MODES)}.")
    typed = transcript_text.strip()
    start = {"patient_name": patient_name.strip(), "mode": mode}
    temp_path = None
    try:
        if typed:
            start["transcript"] = typed
        else:
            if audio_file is None or not (audio_file.filename or "").lower().endswith(".wav"):
                raise HTTPException(400, "Send a mono 16 kHz WAV file (the web page does this for you), or paste a transcript.")
            content = await audio_file.read()
            if not content:
                raise HTTPException(400, "The audio file is empty.")
            with tempfile.NamedTemporaryFile(prefix="consult_", suffix=".wav", delete=False) as f:
                temp_path = f.name
                f.write(content)
            start["audio_path"] = temp_path

        pipeline = MODES[mode]["pipeline"]
        if pipeline == "fast":
            return await _run_fast(mode, start)
        if pipeline == "agentic":
            state = await mode_app.ainvoke(start)
            return {"mode": mode, "transcript": state["transcript"], "extracted": state["extracted"], "triage": None,
                    "guidance": state["guidance"], "evidence": _public_evidence(state.get("evidence", [])),
                    "confidence": state["confidence"], "warnings": state.get("warnings", []),
                    "llm_status": "ok", "agent": state["agent"]}

        state = await agent_app.ainvoke(start)
        return {
            "mode": mode,
            "transcript": state["transcript"],
            "extracted": state["extracted"],
            "prediction": state["prediction"],
            "sheet": state["sheet"],
            "sheet_warnings": state.get("sheet_warnings", []),
            "patient_history": state.get("patient_history", []),
            "similar_cases": state.get("similar_cases", []),
            "trends": state.get("trends", {}),
            "advice": state["advice"],
            "evidence": _public_evidence(state.get("evidence", [])),
            "confidence": state["confidence"],
            "warnings": state.get("warnings", []),
            "agent": state["agent"],
        }
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        print(f"Analysis failed: {type(error).__name__}: {error}")
        raise HTTPException(500, "Analysis failed. Check the server log for details.") from error
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/records")
async def save_record(
    patient_name: str = Form(...),
    diagnosis: str = Form("Not mentioned"),
    medicines: str = Form("[]"),
    vitals: str = Form(""),
    doctor_approach: str = Form(""),
    transcript: str = Form(""),
    dialogue: str = Form("[]"),
    sheet: str = Form("{}"),
    prediction: str = Form("{}"),
    mode: str = Form(DEFAULT_MODE),
):
    """Save a reviewed consultation to SQLite, then index the conversation in the vector database."""
    if not patient_name.strip():
        raise HTTPException(400, "Patient name is required.")
    mode = mode if mode in MODES else DEFAULT_MODE
    med_list = _json_or(medicines, [])
    pred = _json_or(prediction, {})
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO consultations
               (created_at, patient_name, diagnosis, medicines, vitals, doctor_approach, transcript, dialogue, sheet, prediction, mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(timespec="seconds"), patient_name.strip(), diagnosis,
             json.dumps(med_list), vitals, doctor_approach, transcript,
             json.dumps(_json_or(dialogue, [])), json.dumps(_json_or(sheet, {})), json.dumps(pred), mode),
        )
        record_id = cur.lastrowid
    try:
        index_record(record_id, patient_name.strip(), diagnosis, doctor_approach, med_list, pred, transcript, mode)
        indexed = True
    except Exception as error:   # the record is safe in SQLite even if indexing fails; /records/reindex can retry
        print(f"Vector indexing failed for record {record_id}: {error}")
        indexed = False
    return {"id": record_id, "saved": True, "indexed": indexed}


@app.post("/records/reindex")
async def reindex_records():
    """Rebuild the vector index from SQLite (use after upgrading, so old records get conversation chunks)."""
    with db() as conn:
        rows = [row_to_dict(r) for r in conn.execute("SELECT * FROM consultations ORDER BY id")]
    for rec in rows:
        try:
            case_stores[rec.get("mode") or DEFAULT_MODE].delete(where={"record_id": rec["id"]})
        except Exception as error:
            print(f"Vector delete failed for {rec['id']}: {error}")
        index_record(rec["id"], rec["patient_name"], rec["diagnosis"] or "", rec["doctor_approach"] or "",
                     rec["medicines"], rec["prediction"], rec["transcript"] or "", rec.get("mode") or DEFAULT_MODE)
    return {"reindexed": len(rows)}


@app.get("/records")
async def list_records(q: str = "", mode: str = ""):
    like = f"%{q.strip()}%"
    sql = """SELECT * FROM consultations
             WHERE (patient_name LIKE ? OR diagnosis LIKE ? OR medicines LIKE ?)"""
    args = [like, like, like]
    if mode in MODES:
        sql += " AND COALESCE(mode, 'lifestyle') = ?"
        args.append(mode)
    with db() as conn:
        rows = conn.execute(sql + " ORDER BY id DESC", args).fetchall()
    return [row_to_dict(r) for r in rows]


@app.delete("/records/{record_id}")
async def delete_record(record_id: int):
    with db() as conn:
        row = conn.execute("SELECT mode FROM consultations WHERE id=?", (record_id,)).fetchone()
        conn.execute("DELETE FROM consultations WHERE id=?", (record_id,))
    try:
        case_stores[(row["mode"] if row and row["mode"] else DEFAULT_MODE)].delete(where={"record_id": record_id})
    except Exception as error:
        print(f"Vector delete failed: {error}")
    return {"deleted": True}


@app.post("/sheet/pdf")
async def sheet_pdf(
    patient_name: str = Form("Patient"),
    doctor_name: str = Form(""),
    sheet: str = Form("{}"),
):
    """Render the (doctor-reviewed) take-home sheet as a PDF."""
    try:
        data = json.loads(sheet)
    except json.JSONDecodeError as error:
        raise HTTPException(400, "Invalid sheet data.") from error
    return pdf_response(patient_name.strip() or "Patient", doctor_name.strip(), data)


@app.get("/records/{record_id}/sheet.pdf")
async def record_sheet_pdf(record_id: int):
    with db() as conn:
        row = conn.execute("SELECT * FROM consultations WHERE id=?", (record_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Record not found.")
    rec = row_to_dict(row)
    sheet = rec["sheet"] or {
        "diagnosis": rec["diagnosis"],
        "medicines": [{"name": m.get("name"), "dose": m.get("dosage", ""), "how_to_take": "", "duration": ""} for m in rec["medicines"]],
    }
    return pdf_response(rec["patient_name"], "", sheet)