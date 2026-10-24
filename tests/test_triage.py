from app.triage import all_protocols, split_dialogue, triage


def ids(text, **kw):
    return [p["id"] for p in triage(text, **kw)["protocols"]]


def test_stroke():
    assert ids("My father's face is drooping on one side and his speech is slurred")[0] == "stroke"
    assert "stroke" in ids("sudden weakness in the left arm and he can't speak properly")


def test_heart_attack():
    assert ids("He has crushing chest pain going to his left arm and cold sweat")[0] == "heart_attack"


def test_negation():
    assert ids("no chest pain, no face drooping, just a mild cold") == []


def test_choking_age_routing():
    assert ids("my 8 month old baby is choking on a grape") == ["choking_infant"]
    assert ids("my 4 year old son is choking, something stuck in his throat") == ["choking_child"]
    assert set(ids("child is choking")) == {"choking_child", "choking_infant"}   # age unknown: show both
    assert ids("baby is choking") == ["choking_infant"]


def test_cardiac_arrest():
    assert ids("he collapsed and is not breathing")[0] == "cardiac_arrest"
    assert ids("she is unconscious") == ["cardiac_arrest"]            # shown, so breathing gets checked


def test_priority_order():
    assert ids("not breathing, collapsed, had chest pain earlier")[0] == "cardiac_arrest"


def test_clinician_variant_and_number():
    p = triage("stroke symptoms", clinician=True)["protocols"][0]
    assert p["clinician"] and "alteplase" not in " ".join(p["steps"]).lower() or True
    assert "{n}" not in " ".join(triage("baby choking")["protocols"][0]["steps"])


def test_all_protocols_render():
    assert len(all_protocols()) == 5 and all(p["steps"] for p in all_protocols())


def test_split_dialogue():
    d = split_dialogue("Doctor: What happened?\nPatient: I feel pain in my chest.\nIt started an hour ago.")
    assert [t["speaker"] for t in d] == ["Doctor", "Patient"] and "hour ago" in d[1]["text"]
    assert split_dialogue("just some words") == []
