"""Every case must return all four features through the real /analyze endpoint:

  1. possible types + the tests that confirm them
  2. medicines (general ones, and what this clinic gave similar patients)
  3. similar patients from the clinic's own records
  4. diagnosis plan (follow-up for the next meeting + diagnosis considerations)

Each case is run with a model that answers well, answers blank, answers in the wrong shape, and is down.
Run:  python -m pytest tests -q
"""
import asyncio
import json

import pytest

from cases import CASES, T, script_for

MODEL_BEHAVIOURS = ["good", "empty", "garbage", "down"]
BY_ID = {c.id: c for c in CASES}
DIABETES_CASES = [c for c in CASES if c.top]
ASKED = [c for c in CASES if not c.top]

# a confirming test that must be offered for each type (matched with the app's own test key)
CONFIRMING_TESTS = {"t2": {"ogtt", "lipid", "kidney"}, "t1": {"ketone", "antibody", "peptide"}, "pre": {"ogtt"},
                    "gdm": {"ogtt"}, "lada": {"antibody", "peptide"}, "mody": {"peptide", "geneticte"}, "sec": {"reviewofs", "pancreat"}}
# a medicine family the built-in reference must name for each type (checked when the model gives nothing)
REFERENCE_MEDICINE = {"t2": "metformin", "t1": "insulin", "lada": "insulin", "gdm": "insulin", "sec": "insulin", "mody": "sulfonylurea"}

# earlier consultations in the clinic, written the way real doctors write them (different spellings on purpose)
CLINIC_HISTORY = [
    ("Type 2 diabetes", [("Metformin", "500 mg twice daily")]), ("Type II DM", [("Metformin", "1000 mg daily"), ("Glimepiride", "1 mg")]),
    ("Type 1 diabetes", [("Insulin", "10 units")]), ("IDDM", [("Insulin", "12 units")]),
    ("Prediabetes", []), ("Pre-diabetes", [("Metformin", "500 mg")]),
    ("Gestational diabetes", [("Insulin", "6 units")]), ("GDM", [("Insulin", "8 units")]),
    ("LADA", [("Insulin", "8 units")]), ("Latent autoimmune diabetes", [("Insulin", "6 units")]),
    ("MODY 3", [("Gliclazide", "40 mg")]), ("MODY", [("Gliclazide", "80 mg")]),
    ("Secondary diabetes", [("Insulin", "14 units")]), ("Steroid-induced diabetes", [("Insulin", "10 units")]),
]


def save(main, kb, patient, dx, meds, text="Doctor: please start treatment."):
    with main.db() as conn:
        cur = conn.execute(
            "INSERT INTO consultations (created_at,patient_name,diagnosis,medicines,vitals,doctor_approach,transcript,dialogue,sheet,prediction,mode)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-09-01T10:00:00", patient, dx, json.dumps([{"name": n, "dosage": d, "status": "prescribed"} for n, d in meds]),
             "", f"Assessed symptoms and started treatment for {dx}.", text, "[]", "{}", "{}", "lifestyle"))
        rid = cur.lastrowid
    main.index_record(rid, patient, dx, f"Treatment for {dx}", [{"name": n, "dosage": d} for n, d in meds], {}, f"{dx}. {text}", "lifestyle")


@pytest.fixture()
def clinic(main, clean_records):
    import conftest
    for i, (dx, meds) in enumerate(CLINIC_HISTORY):
        save(main, conftest._kb, f"Earlier patient {i}", dx, meds, f"Doctor: diagnosis {dx}")


def analyze(main, llm, case, behaviour, patient="New Patient"):
    llm.mode, llm.script = behaviour, script_for(case)
    return asyncio.run(main.analyze(audio_file=None, patient_name=patient, transcript_text=case.transcript, mode="lifestyle"))


def keys(res):
    return [t["key"] for t in res["prediction"]["possible_types"]]


# ------------------------------------------------------------------------------------ the four features, every case
@pytest.mark.parametrize("behaviour", MODEL_BEHAVIOURS)
@pytest.mark.parametrize("case", DIABETES_CASES, ids=lambda c: c.id)
def test_all_four_features_answer(main, llm, clinic, case, behaviour):
    res = analyze(main, llm, case, behaviour)
    json.dumps(res)                                                     # the page can always receive it

    # 1. possible types and tests
    assert case.top in keys(res), f"{case.top} missing from {keys(res)}"
    if behaviour == "good":
        assert keys(res)[0] == case.top
    tests = {main._test_key(t["test"]) for t in res["prediction"]["tests"]}
    assert "hba1c" in tests, "HbA1c must always be offered"
    assert tests & CONFIRMING_TESTS[case.top], f"no confirming test for {case.top}: {tests}"

    # 2. medicines: general + what this clinic gave
    adv = res["advice"]
    assert adv["possible_medicines"] or adv["standard_medicines"], "no medicine section at all"
    assert adv["standard_medicines"] or adv.get("medicine_note"), "empty medicine list must explain why"
    if behaviour != "good" and case.top in REFERENCE_MEDICINE:
        names = " ".join(m["name"].lower() + " " + m["drug_class"].lower() for m in adv["standard_medicines"])
        assert REFERENCE_MEDICINE[case.top] in names, names
    trends = res["trends"]
    assert trends["records"] >= 1 and (trends["medicines"] or trends["note"]), trends

    # 3. similar patients: found, and all of the same kind of diabetes
    assert res["similar_cases"], "no similar patients returned"
    for rec in res["similar_cases"]:
        assert main.type_key(rec["diagnosis"]) == case.top, f"{rec['diagnosis']} is not similar to {case.top}"

    # 4. diagnosis plan
    assert adv["follow_up"], "no follow-up"
    assert adv["diagnosis_considerations"], "no diagnosis considerations"
    assert adv["diet"] or adv["activity"] or adv["lifestyle"]
    for key in ("follow_up", "diagnosis_considerations", "standard_medicines", "possible_medicines"):
        for item in adv[key]:                                           # nothing unsourced looks verified
            assert item["supported"] or item["note"]


@pytest.mark.parametrize("behaviour", MODEL_BEHAVIOURS)
def test_clinic_medicine_counts_are_per_type(main, llm, clinic, behaviour):
    res = analyze(main, llm, BY_ID["t1_child_dka"], behaviour)
    names = {m["name"].lower() for m in res["trends"]["medicines"]}
    assert names == {"insulin"}, names                                  # no metformin from the type 2 patients
    assert res["trends"]["medicines"][0]["count"] == res["trends"]["records"] == 2


@pytest.mark.parametrize("behaviour", MODEL_BEHAVIOURS)
@pytest.mark.parametrize("case", ASKED, ids=lambda c: c.id)
def test_conversation_without_diabetes_gets_questions_not_guesses(main, llm, clinic, case, behaviour):
    res = analyze(main, llm, case, behaviour)
    assert res["prediction"]["possible_types"] == [] and res["prediction"]["tests"] == []
    assert res["prediction"]["missing_info"], "must say what to ask next"
    assert res["similar_cases"] == [] and res["trends"]["medicines"] == []
    assert res["advice"]["standard_medicines"] == [] and res["advice"]["possible_medicines"] == []
    assert res["advice"]["follow_up"] and res["advice"]["diagnosis_considerations"]
    assert res["trends"]["note"]


# ------------------------------------------------------------------------------------ empty record book
@pytest.mark.parametrize("case", DIABETES_CASES, ids=lambda c: c.id)
def test_new_clinic_with_no_records_still_answers(main, llm, clean_records, case):
    res = analyze(main, llm, case, "good")
    assert res["similar_cases"] == [] and res["trends"]["records"] == 0
    assert "empty" in res["trends"]["note"].lower()
    assert res["advice"]["follow_up"] and res["prediction"]["tests"]


def test_no_match_for_this_type_is_explained(main, llm, clean_records):
    import conftest
    save(main, conftest._kb, "Only patient", "Type 1 diabetes", [("Insulin", "10 units")])
    res = analyze(main, llm, BY_ID["t2_classic"], "good")
    assert res["similar_cases"] == [] and "No saved record matches" in res["trends"]["note"]


def test_records_without_medicines_are_explained(main, llm, clean_records):
    import conftest
    save(main, conftest._kb, "Diet only", "Prediabetes", [])
    res = analyze(main, llm, BY_ID["prediabetes_no_symptoms"], "good")
    assert res["trends"]["records"] == 1 and res["trends"]["medicines"] == []
    assert "none lists a prescribed medicine" in res["trends"]["note"]


def test_own_earlier_visit_is_history_not_a_similar_patient(main, llm, clean_records):
    import conftest
    save(main, conftest._kb, "Asha Rao", "Type 2 diabetes", [("Metformin", "500 mg")])
    save(main, conftest._kb, "Other Person", "Type 2 diabetes", [("Metformin", "500 mg")])
    res = analyze(main, llm, BY_ID["t2_classic"], "good", patient="Asha Rao")
    assert [r["patient_name"] for r in res["patient_history"]] == ["Asha Rao"]
    assert [r["patient_name"] for r in res["similar_cases"]] == ["Other Person"]


# ------------------------------------------------------------------------------------ wording of the diagnosis
@pytest.mark.parametrize("text,key", [
    ("Type 2 diabetes", "t2"), ("Type II diabetes mellitus", "t2"), ("T2DM", "t2"), ("NIDDM", "t2"), ("Type-2 DM", "t2"),
    ("DM2", "t2"), ("Diabetes type 2", "t2"), ("type two diabetes", "t2"),
    ("Type 1 diabetes", "t1"), ("T1DM", "t1"), ("IDDM", "t1"), ("Type I diabetes", "t1"), ("juvenile diabetes", "t1"), ("DM1", "t1"),
    ("Gestational diabetes", "gdm"), ("GDM", "gdm"), ("Prediabetes", "pre"), ("Pre-diabetes", "pre"), ("Impaired glucose tolerance", "pre"),
    ("LADA", "lada"), ("Type 1.5 diabetes", "lada"), ("MODY 3", "mody"), ("Monogenic diabetes", "mody"),
    ("Secondary diabetes", "sec"), ("Steroid-induced diabetes", "sec"), ("Pancreatogenic diabetes", "sec"),
    ("Diabetes mellitus", "und"), ("Diabetes", "und"), ("Diabetes, type to be determined", "und"),
    ("Not mentioned", "other"), ("Hypertension", "other"), ("Another cause (diabetes unlikely)", "other"), ("", "other"),
])
def test_type_key_understands_every_spelling(main, text, key):
    assert main.type_key(text) == key


@pytest.mark.parametrize("label", list(T.values()))
def test_every_type_label_maps_back_to_its_key(main, label):
    assert main.TYPE_LABELS[main.type_key(label)] == label


# ------------------------------------------------------------------------------------ reading the conversation without the model
@pytest.mark.parametrize("case", [c for c in CASES if c.dx], ids=lambda c: c.id)
def test_diagnosis_found_by_rules(main, case):
    found = main.fallback_diagnosis(case.transcript)
    assert main.type_key(found) == main.type_key(case.dx), found


@pytest.mark.parametrize("case", [c for c in CASES if c.meds], ids=lambda c: c.id)
def test_medicines_found_by_rules(main, case):
    got = {m["name"].lower(): m for m in main.fallback_medicines(case.transcript)}
    for name, dose in case.meds:
        assert name.lower() in got and got[name.lower()]["status"] == "prescribed"
        if not dose.lower().startswith("not"):
            assert dose.split()[0] in got[name.lower()]["dosage"], got[name.lower()]


def test_rules_ignore_rejected_diagnosis_and_medicine(main):
    text = "Doctor: This is not type 1 diabetes. Do not take glimepiride. Start metformin 500 mg once daily."
    assert main.fallback_diagnosis(text) == "Not mentioned"
    meds = {m["name"]: m["status"] for m in main.fallback_medicines(text)}
    assert meds == {"Glimepiride": "discussed", "Metformin": "prescribed"}


# ------------------------------------------------------------------------------------ the wider system
@pytest.mark.parametrize("behaviour", MODEL_BEHAVIOURS)
def test_risk_factors_alone_give_prediabetes_questions_and_tests(main, llm, clinic, behaviour):
    case = BY_ID["prediabetes_no_symptoms"]
    res = analyze(main, llm, case, behaviour)
    assert "pre" in keys(res) and res["prediction"]["status"] != "insufficient"
    assert res["prediction"]["tests"] and res["advice"]["follow_up"]


def test_red_flags_reach_the_doctor_without_the_model(main, llm, clinic):
    res = analyze(main, llm, BY_ID["t1_child_dka"], "down")
    assert any("ketoacidosis" in w.lower() for w in res["prediction"]["red_flags"])


def test_warnings_tell_the_doctor_the_model_was_down(main, llm, clinic):
    res = analyze(main, llm, BY_ID["t2_classic"], "down")
    text = " ".join(res["warnings"]).lower()
    assert "unavailable" in text and res["confidence"] == "insufficient"
    assert any("model" in w.lower() for w in res["sheet_warnings"])
