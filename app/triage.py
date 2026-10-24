"""Instant emergency triage. Plain code, no model, no database: answers in milliseconds.

The steps below follow widely used first-aid guidance (AHA / Red Cross / NHS style).
They are the safety net: they are shown even if the language model is slow or offline.
Review them with your clinical lead before real-world use.
"""
import re

from app.modes import EMERGENCY_NUMBER

PROTOCOLS = {
    "cardiac_arrest": {
        "title": "Not responding and not breathing normally",
        "headline": "Possible cardiac arrest. Start CPR now.",
        "domain": "critical_conditions", "metronome": True,
        "signs": ["Does not answer or wake up when shaken", "Not breathing, or only gasping"],
        "steps": [
            "Shout and tap the shoulders. Check breathing for no more than 10 seconds.",
            "Call {n} now and put the phone on speaker. Send someone to fetch a defibrillator (AED).",
            "Lay the person flat on a firm surface. Push hard and fast in the centre of the chest, 100 to 120 times a minute, about 5 to 6 cm deep. Let the chest rise fully each time.",
            "If you are trained, give 2 breaths after every 30 pushes. If not, keep pushing without breaths.",
            "As soon as the AED arrives, switch it on and follow its voice instructions.",
            "Do not stop until the ambulance team takes over or the person starts to breathe normally. Swap with someone every 2 minutes if you can.",
            "Child: push with one or two hands, about one third of the chest depth. Baby: use two fingers.",
        ],
        "avoid": ["Do not wait to see if they recover", "Do not stop to look for a pulse"],
        "tell": ["What happened and when they collapsed", "Known heart disease, medicines", "How long CPR has been going on"],
        "clinician": [
            "Confirm arrest, start high-quality CPR (30:2), attach monitor or defibrillator.",
            "Shockable rhythm (VF or pulseless VT): shock at once and resume CPR for 2 minutes.",
            "IV or IO access. Adrenaline 1 mg every 3 to 5 minutes. Amiodarone 300 mg after the third shock.",
            "Treat reversible causes: hypoxia, hypovolaemia, hypo/hyperkalaemia, hypothermia, tension pneumothorax, tamponade, toxins, thrombosis.",
            "Rotate compressors every 2 minutes. Plan post-arrest care: 12-lead ECG, oxygenation, temperature control, cath lab.",
        ],
    },
    "choking_infant": {
        "title": "Baby under 1 year is choking",
        "headline": "Baby choking: back blows and chest thrusts. Never abdominal thrusts.",
        "domain": "child_choking",
        "signs": ["Cannot cry, cough or breathe", "Silent, mouth open, weak or no cough", "Face turning blue or grey"],
        "steps": [
            "If the baby is coughing hard or crying, let them cough. Stay close and do not hit the back.",
            "If they cannot cry, cough or breathe: shout for help. Have someone call {n}, or put your phone on speaker.",
            "Back blows: sit down. Lay the baby face down along your forearm, resting on your thigh, head lower than the chest. Hold the jaw with your fingers, not the soft throat. Give up to 5 firm blows between the shoulder blades with the heel of your other hand.",
            "Chest thrusts: if still blocked, turn the baby face up along your other forearm, head lower than the body. Put two fingers in the middle of the chest, just below the nipple line. Give up to 5 quick thrusts, about one third of the chest depth.",
            "Look in the mouth. Take out the object only if you can see it. Never sweep blindly with a finger.",
            "Repeat 5 back blows and 5 chest thrusts until the object comes out or the baby cries or breathes.",
            "If the baby goes limp or unresponsive: lay them on a firm surface and start infant CPR (two fingers, 30 pushes then 2 small breaths) while {n} is on speaker.",
            "Even if the object comes out, take the baby to a doctor today.",
        ],
        "avoid": ["No abdominal thrusts on a baby", "No blind finger sweeps", "Do not hold the baby upside down by the feet or shake them", "Do not give water"],
        "tell": ["Baby's age and weight", "What they may have swallowed and when", "Whether they are still breathing or crying"],
    },
    "choking_child": {
        "title": "Child (1 year or older) is choking",
        "headline": "Child choking: 5 back blows, then 5 abdominal thrusts.",
        "domain": "child_choking",
        "signs": ["Cannot speak, cough or breathe", "Clutching the throat, silent, panicked", "Lips or face turning blue"],
        "steps": [
            "If they can cough, speak or cry, encourage them to keep coughing. Do not slap the back.",
            "If they cannot cough, speak or breathe: have someone call {n}, or put your phone on speaker.",
            "Back blows: lean the child forward, support the chest with one hand. Give up to 5 firm blows between the shoulder blades with the heel of your other hand.",
            "Abdominal thrusts: stand or kneel behind the child, arms around the waist. Make a fist just above the belly button, below the ribs, and hold it with your other hand. Give up to 5 sharp thrusts inward and upward.",
            "Alternate 5 back blows and 5 abdominal thrusts until the object comes out or the child can breathe.",
            "If the child becomes unresponsive: lay them on the floor and start CPR. Before each set of breaths, look in the mouth and remove the object only if you can see it.",
            "Always get a child checked at hospital afterwards. Abdominal thrusts can cause internal injury.",
        ],
        "avoid": ["No blind finger sweeps", "Do not give water or food", "Do not hang the child upside down", "Do not hit the back of a child who is coughing well"],
        "tell": ["Child's age and weight", "What they may have swallowed and when", "Whether they are still breathing"],
    },
    "stroke": {
        "title": "Possible stroke",
        "headline": "Possible stroke. Every minute counts. Call now.",
        "domain": "heart_stroke",
        "signs": ["Balance: sudden dizziness or loss of balance", "Eyes: sudden loss of vision or double vision",
                  "Face: one side drooping, uneven smile", "Arm or leg: weakness or numbness on one side",
                  "Speech: slurred, confused or unable to speak", "Time: sudden onset. Call {n} at once"],
        "steps": [
            "Call {n} now. Say \"possible stroke\" and give the exact location.",
            "Note the time the symptoms started, or when the person was last seen normal. Doctors need this to choose treatment.",
            "Keep them calm, sitting or lying with head and shoulders slightly raised. Loosen tight clothing.",
            "Give nothing to eat or drink and no medicines, not even aspirin, unless the emergency doctor tells you to. Some strokes are caused by bleeding, and aspirin can make them worse.",
            "If drowsy or unconscious but breathing, lay them on their side. If they are not breathing normally, start CPR.",
            "Stay with them. Ask the ambulance to take them to a hospital with a CT scanner and a stroke team.",
        ],
        "avoid": ["Do not wait to see if it passes", "No food, water or pills", "Do not drive them yourself if an ambulance is available"],
        "tell": ["Time symptoms began or last seen normal", "Blood thinners and other medicines", "Diabetes, high blood pressure, irregular heartbeat", "Allergies"],
        "clinician": [
            "Establish last-known-well time. Check capillary glucose. ABCs, oxygen only if SpO2 is below 94%.",
            "Activate the stroke code and pre-alert CT. Non-contrast CT head (plus CTA) as fast as possible, target within 20 to 25 minutes of arrival.",
            "Ischaemic and within 4.5 hours of onset, no contraindication: IV thrombolysis, door-to-needle 60 minutes or less.",
            "Large-vessel occlusion: refer for thrombectomy, possible up to 24 hours in selected patients.",
            "Do not lower BP aggressively unless it is above 185/110 before lysis or above 220/120 without lysis. Nil by mouth until a swallow screen.",
        ],
    },
    "heart_attack": {
        "title": "Possible heart attack",
        "headline": "Possible heart attack. Call now. Rest, do not walk.",
        "domain": "critical_conditions",
        "signs": ["Pressure, squeezing or pain in the centre of the chest", "Spreading to the arm, jaw, neck or back",
                  "Short of breath, cold sweat, nausea, light-headed",
                  "Women, older people and people with diabetes may only feel breathless, tired or have indigestion"],
        "steps": [
            "Call {n} now. Do not wait for the pain to pass.",
            "Sit them down and keep them still, half-sitting with knees bent. Loosen tight clothing.",
            "If the emergency operator or a doctor agrees, and they are not allergic to aspirin and have no bleeding problem, let them chew one regular adult aspirin (300 to 325 mg). Chew it, do not swallow it whole.",
            "If a doctor has prescribed glyceryl trinitrate (nitroglycerin) for them, help them use it exactly as prescribed.",
            "Unlock the door, send someone to meet the ambulance, and fetch an AED if there is one nearby.",
            "Stay with them. If they stop responding and are not breathing normally, start CPR and use the AED.",
        ],
        "avoid": ["Do not let them walk, climb stairs or drive", "No food or drink", "Do not give aspirin for a suspected stroke"],
        "tell": ["When the pain started", "Heart disease, diabetes, blood pressure", "Medicines, especially blood thinners", "Allergies"],
        "clinician": [
            "12-lead ECG within 10 minutes of arrival. Attach monitor and defibrillator pads, IV access.",
            "STEMI: activate the cath lab. Primary PCI with door-to-balloon 90 minutes or less (120 if transferred). If PCI cannot happen within 120 minutes, give fibrinolysis within 30 minutes of arrival.",
            "Aspirin 162 to 325 mg chewed unless contraindicated. Nitrates only if not hypotensive and no RV infarct or PDE-5 inhibitor use. Oxygen only if SpO2 is below 90%.",
            "Serial high-sensitivity troponin if ECG is non-diagnostic. Consider dissection or PE before anticoagulating.",
        ],
    },
}

_Q = r"(sudden\w*|suddenly)"
SIGNS = {
    "cardiac_arrest": [
        r"cardiac arrest", r"no pulse", r"(not|isn'?t|stopped|stop|barely|no longer) breath\w*", r"gasping",
        r"(unresponsive|unconscious|collapsed|not responding|won'?t wake|not waking|passed out)",
    ],
    "choking": [
        r"chok\w*", r"stuck (in|down) (his|her|the|my|their|its) (throat|windpipe)",
        r"swallowed (a |an |some |the )?(toy|coin|button|battery|nut|peanut|grape|candy|sweet|bead|marble|magnet|piece|food|bone)",
        r"(food|drink|milk) (went|gone|going) down the wrong",
    ],

    "stroke": [   # FAST signs (face, arm, speech, time) plus sudden vision / balance loss. Have your clinical lead review these.
        r"\bstroke\b", r"face\b.{0,25}(droop\w*|numb\w*|uneven|lopsided|crooked|sagging)", r"(droop\w*|sagging)\b.{0,20}face",
        r"slurr\w*", r"speech\b.{0,15}(garbled|confus\w*|strange)", r"(can'?t|cannot|unable to|trouble|difficulty) (speak|talk|speaking|talking|understand)\w*",
        r"sudden\w*.{0,30}(weakness|numbness).{0,30}(arm|leg|face|one side|side)",
        r"(arm|leg)\b.{0,20}(weak\w*|numb\w*)\b.{0,20}(sudden|one side)", r"(one|left|right) side.{0,20}(weak\w*|numb\w*|paraly\w*)",
        r"sudden\w*.{0,25}(loss of (vision|balance)|double vision|blind\w*)", r"worst headache",
    ],
    "heart_attack": [
        r"heart attack", r"chest\b.{0,12}(pain|pressure|tight\w*|discomfort|heav\w*|squeez\w*|crushing)",
        r"(pain|pressure|tightness|squeezing) (in|across|on|around) (the |my |his |her )?chest",
        r"pain\b.{0,30}(left arm|jaw|shoulder|neck)", r"(cold sweat|sweating).{0,40}(chest|breath)",
    ],
}
NEG = re.compile(r"(\bno\b|\bnot\b|\bnever\b|n't|\bdenies\b|\bwithout\b|\bnone\b)[^.?!]{0,20}$")
# these phrases contain "not"/"no" themselves, so negation must not cancel them
NEG_PROOF = re.compile(r"(not|isn'?t|stopped|stop|barely|no longer) breath|no pulse|not respond|not wak")


def _hit(patterns, text):
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            if NEG_PROOF.search(m.group(0)) or not NEG.search(text[max(0, m.start() - 25):m.start()]):
                found.append(m.group(0).strip())
                break
    return found


def _child_age_group(text: str):
    """'infant', 'child' or None (unknown)."""
    m = re.search(r"(\d+)\s*[- ]?(day|week|month|mo)s?\b", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return "infant" if unit in ("day", "week") or n < 12 else "child"
    m = re.search(r"(\d+)\s*[- ]?(year|yr)s?\b", text)
    if m:
        return "infant" if int(m.group(1)) < 1 else "child"
    if re.search(r"\b(baby|infant|newborn|under (a|one|1) year|few months)\b", text):
        return "infant"
    if re.search(r"\b(toddler|adult|man|woman|grandfather|grandmother|grandpa|grandma)\b", text):
        return "child"   # same back-blow / abdominal-thrust method
    return None          # "child", "kid", "my son"... could be under 1 year: the caller shows both cards


def render(pid: str, clinician: bool = False) -> dict:
    p = PROTOCOLS[pid]
    steps = p["clinician"] if clinician and p.get("clinician") else p["steps"]
    fmt = lambda items: [s.replace("{n}", EMERGENCY_NUMBER) for s in items]
    return {"id": pid, "title": p["title"], "headline": p["headline"], "domain": p["domain"],
            "signs": fmt(p["signs"]), "steps": fmt(steps), "avoid": fmt(p["avoid"]), "tell": fmt(p["tell"]),
            "metronome": bool(p.get("metronome")), "clinician": bool(clinician and p.get("clinician"))}


def triage(text: str, clinician: bool = False) -> dict:
    """Which built-in protocols match this text? Always fast, never raises."""
    t = (text or "").lower()
    hits = {k: _hit(v, t) for k, v in SIGNS.items()}
    ids = []
    arrest = hits["cardiac_arrest"]
    # "unconscious" alone could be a faint: only call it arrest with a breathing sign or the words cardiac arrest
    if arrest and (len(arrest) >= 2 or any(re.search(r"breath|pulse|cardiac|gasp", a) for a in arrest)):
        ids.append("cardiac_arrest")
    if hits["choking"]:
        age = _child_age_group(t)
        ids += ["choking_infant"] if age == "infant" else ["choking_child"] if age == "child" else ["choking_child", "choking_infant"]

    if hits["stroke"]:
        ids.append("stroke")
    if hits["heart_attack"]:
        ids.append("heart_attack")
    if arrest and "cardiac_arrest" not in ids:
        ids.append("cardiac_arrest")   # collapsed / unresponsive: still show the CPR card, last
    matched = {k: v for k, v in hits.items() if v}
    return {"emergency_number": EMERGENCY_NUMBER, "matched_signs": matched,
            "protocols": [render(i, clinician) for i in dict.fromkeys(ids)]}


def all_protocols(clinician: bool = False) -> list:
    return [render(i, clinician) for i in ("stroke", "heart_attack", "cardiac_arrest", "choking_infant", "choking_child")]


def split_dialogue(text: str) -> list:
    """Turn 'Doctor: ... Patient: ...' typed text into turns. Returns [] when nobody is labelled."""
    turns = []
    for line in re.split(r"\n+|(?<=[.?!])\s+(?=(?:Doctor|Dr|Patient|Parent|Caller|Nurse)\s*:)", text or ""):
        m = re.match(r"\s*(Doctor|Dr\.?|Patient|Parent|Caller|Nurse)\s*:\s*(.+)", line, re.I)
        if m:
            who = "Doctor" if m.group(1).lower().startswith(("doc", "dr", "nurse")) else "Patient"
            turns.append({"speaker": who, "text": m.group(2).strip()})
        elif turns and line.strip():
            turns[-1]["text"] += " " + line.strip()
    return turns
