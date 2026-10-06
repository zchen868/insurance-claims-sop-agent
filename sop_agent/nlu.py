"""Caller-understanding layer. Turns one caller utterance into a structured
extraction. Claude does it when a key is configured; a rule-based extractor
always runs too, so the demo works offline and deterministic fields
(email, phone, policy number, dates) are double-checked."""
from __future__ import annotations

import json
import re

INTENTS = ["claim_status", "denial_question", "document_submission", "next_steps",
           "payment_question", "appeal", "general_claim_question", "list_claims",
           "new_claim", "account_change", "coverage_question", "email_summary", "none"]
EMOTIONS = ["neutral", "frustrated", "angry", "anxious", "confused", "sad", "positive"]


def _nullable(t, enum=None):
    s = {"type": t}
    if enum:
        s["enum"] = enum
    return {"anyOf": [s, {"type": "null"}]}


def _obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


SCHEMA = _obj({
    "identity": _obj({
        "full_name": _nullable("string"),
        "dob": _nullable("string"),
        "phone": _nullable("string"),
        "email": _nullable("string"),
        "id_last4": _nullable("string"),
        "policy_number": _nullable("string"),
    }),
    "caller_role": {"type": "string", "enum": ["policyholder", "representative", "unknown"]},
    "rep_name": _nullable("string"),
    "rep_relationship": _nullable("string"),
    "intent": {"type": "string", "enum": INTENTS},
    "case_hints": _obj({
        "case_id": _nullable("string"),
        "case_type": _nullable("string", ["healthcare", "dental", "auto"]),
        "status": _nullable("string", ["denied", "open", "closed"]),
        "month": _nullable("integer"),
        "year": _nullable("integer"),
    }),
    "question": _nullable("string"),
    "scope": {"type": "string", "enum": ["in_scope", "out_of_scope", "smalltalk"]},
    "emotion": {"type": "string", "enum": EMOTIONS},
    "refuses_to_verify": {"type": "boolean"},
    "wants_human": {"type": "boolean"},
    "confirmation": _nullable("string", ["yes", "no"]),
    "email_choice": _nullable("string", ["send", "skip"]),
    "done": {"type": "boolean"},
    "switch_case": {"type": "boolean"},
    "additional_case_hints": {"type": "array", "items": _obj({
        "case_id": _nullable("string"),
        "case_type": _nullable("string", ["healthcare", "dental", "auto"]),
        "status": _nullable("string", ["denied", "open", "closed"]),
        "month": _nullable("integer"),
        "year": _nullable("integer"),
    })},
    "safety_concern": {"type": "boolean"},
    "preferred_name": _nullable("string"),
})

SYSTEM = """You are the language-understanding component of an insurance claims support line.
Extract structured data from the caller's latest message. You do not reply to the caller.

Rules:
- identity: only values the caller states in THIS message about the policyholder. Normalize dob to YYYY-MM-DD,
  phone to digits (keep a leading +1 if given), id_last4 to 4 digits (SSN or national ID last four),
  policy_number uppercase like POL-9921. If a representative gives their own name, put it in rep_name, not full_name.
- caller_role: "representative" if they call on behalf of someone else (e.g. "calling for my mother").
- case_hints: anything that identifies which claim they mean (type, status, month/year of the claim, claim id like CL-2048).
  Resolve relative months against today's date given below.
- intent: their goal. denial_question = why denied / what is missing; document_submission = how/where/format of sending docs;
  next_steps = what to do now, deadlines, timing; claim_status = status of a claim; payment_question = amounts/payments;
  list_claims = wants to know which claims they have / see all their claims ("do I have any claims?", "show my claims");
  new_claim = wants to file/open/submit a NEW claim; account_change = wants to change contact details, address, name,
  beneficiary or policy settings, or cancel the policy; coverage_question = asks what their policy covers / benefits /
  eligibility; email_summary = asks to have a summary / recap of this conversation emailed.
- additional_case_hints: if the message ALSO asks about a second, different claim, its hints go here (else []).
- preferred_name: only if the caller says how they want to be addressed ("call me Maggie", "I go by Tian"), else null.
- safety_concern: true if the caller expresses thoughts of suicide, self-harm, or being in danger.
- question: the caller's concrete question restated in one sentence, or null.
- scope: questions asking you to interpret medical results or diagnoses ("does this report mean I have cancer?") are
  out_of_scope even though they mention claim documents. in_scope = insurance claims / policy / this call (incl. identity details, greetings that move the call forward);
  smalltalk = greetings, thanks, pleasantries; out_of_scope = unrelated topics (tech, trivia, general knowledge, coding, other companies).
- emotion: the caller's emotional state in this message (neutral unless there are clear signs of feeling;
  merely asking a question, even an off-topic one, is neutral).
- refuses_to_verify: the caller declines or pushes back on giving identity details.
- wants_human: explicitly asks for a human/agent/supervisor/representative.
- confirmation: yes/no if the message answers the agent's last yes/no question.
- email_choice: "send" or "skip" if the message answers the offer to email a summary.
- done: the caller indicates they have no more questions (e.g. "that's all", "no thanks, bye").
- switch_case: they want to talk about a different claim than the one being discussed."""


def llm_extract(llm, utterance, phase, last_agent, today):
    user = json.dumps({"today": today, "current_phase": phase, "agent_last_message": last_agent,
                       "caller_message": utterance})
    return llm.json(SYSTEM, user, SCHEMA)


# ---------------- rule-based extractor ----------------
STOP_NAMES = {"Hi", "Hello", "Hey", "Thanks", "Thank", "Fine", "Sure", "Ok", "Okay", "Yes", "No", "My", "The", "This",
              "Calling", "Just", "Please", "What", "Why", "How", "I", "Is", "Can", "Policy", "Dental", "Auto", "Healthcare"}
NAME_STOP = {"calling", "and", "the", "a", "an", "policyholder", "here", "on", "for", "about", "with", "from", "not",
             "just", "trying", "looking", "his", "her", "my", "so", "very", "really", "still", "having", "going", "born",
             "dob", "phone", "email", "ssn", "policy", "at", "in", "to"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTH_RE = "|".join(MONTHS)
OUT_OF_SCOPE = re.compile(
    r"\b(rl|reinforcement learning|machine learning|python|javascript|code|recipe|weather|stock|bitcoin|crypto|"
    r"movie|song|football|basketball|world cup|soccer|capital of|president|joke|poem|math|homework|translate|chatgpt|llm|"
    r"transformers?|deep learning|neural network|artificial intelligence|"
    r"what is (?!my|the status|a deductible|an? (?:appeal|claim|copay|deductible))|who is|tell me about (?!my))",
    re.I)
# Medical interpretation is out of scope even though it mentions claim documents.
MEDICAL = re.compile(r"\b((do|does|did) .{0,40}mean (that )?i have|do i have (cancer|a tumou?r)|is it (cancer|serious|malignant)|"
                     r"what does my (diagnosis|biopsy|result) mean)", re.I)
IN_SCOPE = re.compile(r"\b(claim|policy|insur|denied|denial|appeal|document|report|note|upload|submit|payment|paid|"
                      r"reimburse|deductible|coverage|verify|ssn|dob|birth|email|phone|status|cl-\d+|pol-\d+|"
                      r"estimate|photo|portal|deadline|fax|mail)", re.I)
NEG = {
    "angry": re.compile(r"\b(ridiculous|unacceptable|furious|angry|stupid|useless|wtf|damn|hell|absurd|outrageous)\b|!!|f\*+|\bfuck", re.I),
    "frustrated": re.compile(r"\b(already told|i told you|again\??|frustrat|annoy|waste of time|just tell me|come on|seriously)\b", re.I),
    "anxious": re.compile(r"\b(worried|anxious|scared|afraid|stress|nervous|panic|can't afford|cannot afford)\b", re.I),
    "confused": re.compile(r"\b(confus|don't understand|do not understand|what do you mean|not sure what)\b", re.I),
}


def rule_extract(utterance, phase, last_agent, today=None):
    t = utterance.strip()
    lo = t.lower()
    ident = {k: None for k in ("full_name", "dob", "phone", "email", "id_last4", "policy_number")}

    if m := re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", t):
        ident["email"] = m.group(0).rstrip(".")
    if m := re.search(r"\bpol[\s-]?(\d{3,})\b", t, re.I):
        ident["policy_number"] = f"POL-{m.group(1)}"
    if m := re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", t):
        ident["dob"] = m.group(0)
    elif m := re.search(rf"\b({MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b", lo):
        ident["dob"] = f"{m.group(3)}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"
    elif m := re.search(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b", t):
        ident["dob"] = f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    if m := re.search(r"(?:\+?1[\s.-]?)?\(?\b(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})\b", t):
        ident["phone"] = "".join(m.groups())
    if m := re.search(r"\b(?:ssn|social|last\s*(?:four|4)|national id|id)[^0-9]{0,25}(\d{4})\b", lo):
        ident["id_last4"] = m.group(1)
    elif (last_agent or "").lower().find("last 4") >= 0 and (m := re.fullmatch(r"\D*(\d{4})\D*", t)):
        ident["id_last4"] = m.group(1)
    name_m = re.search(r"\b(?:my name is|this is|i am|i'm|name's|name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", t)
    if not name_m:  # lowercase typing: "hi i am ya wen li"
        lm = re.search(r"\b(?:my name is|this is|i am|i'm|im|name's|name is)\s+([a-z]+(?:\s+[a-z]+){1,2})\b", t, re.I)
        if lm:
            words = []
            for w in lm.group(1).split():
                if w.lower() in NAME_STOP:
                    break
                words.append(w)
            if len(words) >= 2:
                name_m = re.match(r"(.*)", " ".join(w.capitalize() for w in words))
    rep_name = rep_rel = None
    role = "unknown"
    rel_m = re.search(r"\b(?:on behalf of|for)\s+my\s+(mother|mom|father|dad|wife|husband|son|daughter|parent)\b", lo)
    if rel_m or re.search(r"\bon behalf of\b", lo):
        role = "representative"
        rep_rel = {"mom": "son/daughter", "mother": "son/daughter"}.get(rel_m.group(1), None) if rel_m else None
        if name_m:
            rep_name = name_m.group(1)
        if m := re.search(r"\b(?:mother|mom|father|dad|wife|husband|son|daughter|parent)(?:'s name is|,| is| named)?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", t):
            ident["full_name"] = m.group(1)
    else:
        if name_m and name_m.group(1).split()[0] not in ("Calling", "Not", "The", "Just"):
            ident["full_name"] = name_m.group(1)
        if not ident["full_name"] and (m := re.match(r"\s*([a-z]+(?:\s+[a-z]+){1,2})\s*(?:,|(?=\s+\d))", t, re.I)) \
                and not ({w.lower() for w in m.group(1).split()} & NAME_STOP) and m.group(1).split()[0].capitalize() not in STOP_NAMES:
            ident["full_name"] = " ".join(w.capitalize() for w in m.group(1).split())
        if not ident["full_name"] and (m := re.search(
                r"(?:^|[.,!:;]\s*)(?:(?:it'?s|name|full name)[:\s]+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?=\s*(?:,|\.|$|born|dob|\bd\.?o\.?b))", t)):
            if m.group(1).split()[0] not in STOP_NAMES:
                ident["full_name"] = m.group(1)
        if re.search(r"\b(i'?m|i am) the policy ?holder\b", lo):
            role = "policyholder"

    hints = {"case_id": None, "case_type": None, "status": None, "month": None, "year": None}
    if m := re.search(r"\bcl-?(\d{3,})\b", lo):
        hints["case_id"] = f"CL-{m.group(1)}"
    for word, ct in (("health", "healthcare"), ("medical", "healthcare"), ("hospital", "healthcare"),
                     ("dental", "dental"), ("dentist", "dental"), ("auto", "auto"), ("car ", "auto"),
                     ("vehicle", "auto"), ("accident", "auto")):
        if word in lo + " ":
            hints["case_type"] = ct
            break
    if re.search(r"\b(denied|denial|rejected)\b", lo):
        hints["status"] = "denied"
    elif re.search(r"\b(open|in progress|pending)\b claim", lo):
        hints["status"] = "open"
    elif re.search(r"\b(closed|settled)\b", lo):
        hints["status"] = "closed"
    dob_span = ident["dob"] and re.search(rf"({MONTH_RE})\s+\d", lo)
    for mname, mi in MONTHS.items():
        if re.search(rf"\b(from|in|of|since|back in)\s+{mname}\b", lo) and not (dob_span and dob_span.group(1) == mname):
            hints["month"] = mi
            break
    if not ident["dob"] and (ym := re.search(r"\b(?:from|in|of|filed in)\s+((?:19|20)\d{2})\b", lo)):
        hints["year"] = int(ym.group(1))
    if today and re.search(r"\blast month\b", lo):
        y, mth = int(today[:4]), int(today[5:7]) - 1
        hints["month"], hints["year"] = (mth, y) if mth else (12, y - 1)

    intent = "none"
    if re.search(r"\b(why|reason).{0,30}(denied|denial|rejected)|denied|denial|what.{0,20}missing", lo):
        intent = "denial_question"
    if re.search(r"\b(how|where).{0,20}(submit|send|upload)|format|pdf|scan|portal|fax", lo):
        intent = "document_submission"
    elif re.search(r"\b(next step|what (do|should) i do|deadline|how long|how soon|when (do|should) i)", lo):
        intent = "next_steps"
    elif re.search(r"\b(status|update on)\b", lo) and intent == "none":
        intent = "claim_status"
    elif re.search(r"\b(pay|paid|payment|reimburs|amount|how much)", lo):
        intent = "payment_question"
    elif re.search(r"\bappeal\b", lo):
        intent = "appeal"
    if re.search(r"\b(file|open|start|submit|make|create)\s+(a\s+)?new\s+claim\b|\bnew claim\b", lo):
        intent = "new_claim"
    elif re.search(r"\b(change|update|correct)\s+(my\s+)?(email|address|phone|name|beneficiary|contact)", lo):
        intent = "account_change"
    elif re.search(r"\b(what does my policy cover|am i covered|is .{0,30} covered|coverage for|my benefits)\b", lo):
        intent = "coverage_question"
    if re.search(r"\b(email|send)( me)? (a |the )?(summary|recap)|email (me )?(this|that)\b", lo):
        intent = "email_summary"
    if intent == "none" and re.search(r"\b(my claims|any claims?|list (of )?(my )?claims|what claims|which claims|all (of )?my claims|show me my claims)\b", lo):
        intent = "list_claims"

    emotion = "neutral"
    for emo, rx in NEG.items():
        if rx.search(t):
            emotion = emo
            break

    wants_human = bool(re.search(r"\b(human|real person|agent|representative|supervisor|manager|someone else)\b", lo)) \
        and not re.search(r"\bi'?m (the|a) representative\b", lo) and role != "representative"
    refuses = bool(re.search(r"\b(not giving|won't give|will not give|don't want to (give|share)|not going to (give|tell)|"
                             r"none of your business|why do you need|refuse)\b", lo)) or \
        (phase == "VERIFY_ID" and re.fullmatch(r"\W*(i said )?no+\W*", lo) is not None) or \
        (phase == "VERIFY_ID" and emotion in ("angry", "frustrated") and re.search(r"already told|just tell me|give me my", lo) is not None)

    yes = re.search(r"^\s*(yes|yeah|yep|sure|correct|right|that's (it|right|the one)|please do|ok(ay)?|go ahead)\b", lo)
    no = re.search(r"^\s*(no|nope|not really|nah)\b", lo)
    confirmation = "yes" if yes else "no" if no else None
    email_choice = None
    if re.search(r"\b(send|email it|email me|yes please)\b", lo) and not re.search(r"\b(don'?t|do not|no)\b.{0,10}send", lo):
        email_choice = "send"
    elif re.search(r"\b(skip|no thanks|no thank you|don'?t send|do not send|not necessary|no need)\b", lo):
        email_choice = "skip"
    elif "email" in (last_agent or "").lower() and confirmation:
        email_choice = "send" if confirmation == "yes" else "skip"
    done = bool(re.search(r"\b(that'?s all|that is all|nothing else|no more questions|i'?m good|bye|that'?s it|all set)\b", lo)) \
        or (confirmation == "no" and "anything else" in (last_agent or "").lower())

    has_identity = any(v for v in ident.values())
    if MEDICAL.search(lo) or (OUT_OF_SCOPE.search(lo) and not IN_SCOPE.search(lo) and not has_identity):
        scope = "out_of_scope"
    elif re.fullmatch(r"\W*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|ok|okay)\W*", lo):
        scope = "smalltalk"
    else:
        scope = "in_scope"

    question = t if ("?" in t and scope == "in_scope") else None
    return {
        "identity": ident, "caller_role": role, "rep_name": rep_name, "rep_relationship": rep_rel,
        "intent": intent, "case_hints": hints, "question": question, "scope": scope, "emotion": emotion,
        "refuses_to_verify": refuses, "wants_human": wants_human, "confirmation": confirmation,
        "email_choice": email_choice, "done": done,
        "switch_case": bool(re.search(r"\b(other|another|different) claim\b", lo)),
        "additional_case_hints": [],
        "medical": bool(MEDICAL.search(lo)),
        "preferred_name": (lambda mm: mm.group(1).capitalize() if mm else None)(
            re.search(r"\b(?:call me|i go by|please call me)\s+([A-Za-z][a-z]+)", t, re.I)),
        "safety_concern": bool(re.search(r"\b(kill myself|suicid|end my life|don'?t want to (be alive|live)|hurt myself|"
                                         r"no reason to live|better off dead)", lo)),
    }


def extract(llm, utterance, phase, last_agent, today):
    """Claude extraction when available, with rule-based values filling any gaps
    in the deterministic identity fields."""
    rules = rule_extract(utterance, phase, last_agent, today)
    if llm is None:
        return _sanitize(rules, utterance), "rules"
    out = llm_extract(llm, utterance, phase, last_agent, today)
    if out is None or not isinstance(out, dict):
        return _sanitize(rules, utterance), "rules (LLM error: %s)" % llm.last_error
    # Providers without strict schema enforcement may omit keys: fill from the rules.
    for k, v in rules.items():
        if not isinstance(out.get(k), type(v)) and v is not None:
            out[k] = v
        out.setdefault(k, v)
    out["identity"] = {**{k: None for k in rules["identity"]}, **(out.get("identity") or {})}
    out["case_hints"] = {**{k: None for k in rules["case_hints"]}, **(out.get("case_hints") or {})}
    for k in ("email", "policy_number", "dob", "phone", "id_last4"):
        if not out["identity"].get(k) and rules["identity"].get(k):
            out["identity"][k] = rules["identity"][k]
    # Deterministic backstop: medical interpretation is never answered, whatever the model decided.
    out["medical"] = rules["medical"]
    # ... and an explicit request to see one's claims always lists them.
    if rules["intent"] == "list_claims" and out.get("intent") in ("none", "general_claim_question", "claim_status", None):
        out["intent"] = "list_claims"
    return _sanitize(out, utterance), "llm"


def _sanitize(x, utterance):
    """Drop an ID last-4 that only appears inside a claim number (e.g. the 3001 in CL-3001)."""
    last4 = (x.get("identity") or {}).get("id_last4")
    if last4 and last4 not in re.sub(r"cl-?\d+", " ", utterance, flags=re.I):
        x["identity"]["id_last4"] = None
    return x
