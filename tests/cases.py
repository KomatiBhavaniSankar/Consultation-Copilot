"""Test conversations covering every diabetes situation the copilot must handle.

Each case has: the conversation, the diagnosis the doctor says (or None), the type key the screening should
rank first, and what a good model would answer. The model answers are scripted so the TESTS check the
pipeline and the fall-backs, not a particular model's mood.
"""
from dataclasses import dataclass, field

T = {"t2": "Type 2 diabetes", "t1": "Type 1 diabetes", "pre": "Prediabetes", "gdm": "Gestational diabetes",
     "lada": "LADA (latent autoimmune diabetes in adults)", "mody": "MODY (monogenic diabetes)",
     "sec": "Secondary diabetes (drug, pancreatic or endocrine cause)"}


@dataclass
class Case:
    id: str
    transcript: str
    dx: str | None            # diagnosis spoken by the doctor, None = not stated
    top: str | None           # type key that should be ranked first, None = not enough information
    symptoms: list = field(default_factory=list)   # symptom names a good model returns
    meds: list = field(default_factory=list)       # (name, dose) the doctor prescribed


CASES = [
    Case("t2_classic",
         "Doctor: What brings you in? Patient: I am 52. I am always thirsty and I pass urine a lot, even at night. I feel tired "
         "and I have gained weight. My father had diabetes. Doctor: This looks like type 2 diabetes. Start metformin 500 mg twice daily "
         "and walk 30 minutes every day.",
         "Type 2 diabetes", "t2", ["Excess thirst", "Frequent urination", "Tiredness"], [("Metformin", "500 mg twice daily")]),
    Case("t1_child_dka",
         "Doctor: Tell me what happened. Patient: I am 14. I lost weight very fast, I am very thirsty, and I have been vomiting. "
         "My mother says my breath smells fruity and I am drowsy. Doctor: I suspect type 1 diabetes. Check ketones now and start insulin.",
         "Type 1 diabetes", "t1", ["Unexplained weight loss", "Excess thirst"], [("Insulin", "Not mentioned")]),
    Case("prediabetes_no_symptoms",
         "Doctor: Any complaints? Patient: No, none. I just came for a check. I have a desk job and I do not exercise. My mother has diabetes. "
         "Doctor: Your fasting sugar is 108, that is prediabetes. Lose weight and recheck in 3 months.",
         "Prediabetes", "pre", [], []),
    Case("gestational",
         "Doctor: How far along are you? Patient: I am 29 weeks pregnant. I am very thirsty and urinating more than usual. "
         "Doctor: This could be gestational diabetes. We will do a glucose tolerance test and start diet control.",
         "Gestational diabetes", "gdm", ["Excess thirst", "Frequent urination"], []),
    Case("lada_lean_adult",
         "Doctor: What is the problem? Patient: I am 38 and slim. Over six months I lost weight and I am tired all the time and thirsty. "
         "The tablets from my old doctor stopped working. Doctor: This may be LADA. We will send antibody tests and start insulin 10 units at night.",
         "LADA", "lada", ["Unexplained weight loss", "Tiredness", "Excess thirst"], [("Insulin", "10 units at night")]),
    Case("mody_family",
         "Doctor: Tell me about your family. Patient: I am 19. My father, my aunt and my grandfather all got diabetes in their twenties. "
         "I feel fine, only slightly thirsty sometimes. Doctor: A MODY gene test is worth doing.",
         "MODY", "mody", ["Excess thirst"], []),
    Case("steroid_induced",
         "Doctor: Are you on any medicines? Patient: Yes, prednisone for my arthritis for a year. I have become very thirsty and I urinate often. "
         "Doctor: The steroid may be raising your sugar, this is secondary diabetes. Review the steroid dose.",
         "Secondary diabetes", "sec", ["Excess thirst", "Frequent urination"], []),
    Case("diagnosis_not_stated",
         "Doctor: What is troubling you? Patient: I am 47. I am always thirsty, urinating a lot, my vision is blurred and my wounds heal slowly. "
         "I am overweight and my brother has sugar. Doctor: Let us get some blood tests first.",
         None, "t2", ["Excess thirst", "Frequent urination", "Blurred vision"], []),
    Case("roman_numeral_dx",
         "Doctor: Your reports are back. Patient: Is it serious? Doctor: You have Type II diabetes mellitus. Take glimepiride 1 mg every morning. "
         "Patient: I am always thirsty and tired.",
         "Type II diabetes mellitus", "t2", ["Excess thirst", "Tiredness"], [("Glimepiride", "1 mg every morning")]),
    Case("abbreviation_dx",
         "Doctor: You have T2DM. Patient: I have tingling in my feet and I am tired. Doctor: Start metformin 500 mg once daily.",
         "T2DM", "t2", ["Tingling or numbness", "Tiredness"], [("Metformin", "500 mg once daily")]),
    Case("vague_conversation",
         "Doctor: How are you? Patient: I have a headache since morning. Doctor: Take rest and drink water.",
         None, None, [], []),
]


def extract_for(c: Case) -> dict:
    return {"dialogue": [], "diagnosis": c.dx or "Not mentioned",
            "medicines": [{"name": n, "dosage": d, "status": "prescribed"} for n, d in c.meds],
            "vitals": "Not mentioned", "doctor_approach": "Assessed symptoms and risk factors, ordered tests, started treatment."}


def predict_for(c: Case) -> dict:
    if c.top is None:
        return {"ready": False, "summary": "", "symptoms": [], "possible_types": [], "tests": [], "missing_info": ["Ask about thirst, urination, weight"]}
    return {"ready": True, "summary": "Symptoms suggest diabetes.",
            "symptoms": [{"symptom": s, "quote": ""} for s in c.symptoms],
            "routine": {"diet": "Not mentioned", "activity": "Not mentioned", "sleep": "Not mentioned",
                        "family_history": "Not mentioned", "other": "Not mentioned"},
            "possible_types": [{"type": T[c.top], "likelihood": "high", "reasons": ["fits history"], "against": []}],
            "tests": [{"test": "HbA1c", "why": "confirms diagnosis"}], "missing_info": [], "red_flags": []}


def advice_for(c: Case) -> dict:
    if c.top is None:     # a careful model says nothing clinical about a conversation that is not about diabetes
        return {}
    return {"standard_medicines": [{"name": "Metformin", "drug_class": "Biguanide", "why": "first line", "evidence": ["E1"]}],
            "possible_medicines": [{"name": "Insulin", "drug_class": "Insulin", "why": "if needed", "evidence": []}],
            "diet": [{"text": "Cut sugary drinks", "evidence": []}], "activity": [{"text": "Walk 30 minutes daily", "evidence": []}],
            "lifestyle": [{"text": "Check feet daily", "evidence": []}],
            "follow_up": [{"text": "Review HbA1c in 3 months", "evidence": []}],
            "diagnosis_considerations": [{"text": "Check kidney function", "evidence": []}]}


def sheet_for(c: Case) -> dict:
    return {"diagnosis": c.dx or "Not stated", "summary": "Plan explained.", "medicines": [],
            "lifestyle": {}, "follow_up": "Not stated"}


def script_for(c: Case) -> dict:
    return {"extract": extract_for(c), "predict": predict_for(c), "advice": advice_for(c), "sheet": sheet_for(c)}
