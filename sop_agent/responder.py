"""Reply layer. Claude phrases the engine's plan naturally; deterministic
templates render the same plan when no model is configured or when the
model's draft fails the output guard (leak / ungrounded fact checks)."""
from __future__ import annotations

import json
import re
from datetime import date

from .data import BACKEND

SYSTEM = """You are Ava, a warm, professional customer-service agent on an insurance company's claims support chat.
You speak in plain conversational language, like a skilled human representative.

You are given a PLAN produced by the company's SOP engine. Write the single reply that performs every act in the
plan, in order, woven into natural conversation. The plan is authoritative:
- Use ONLY facts that appear in the plan (claim data, guidance text, masked contact details). Never invent or estimate
  amounts, dates, document names, deadlines, policies, timelines or outcomes. If the facts do not answer the
  caller's question, say you don't have that information and offer a human representative.
- When "gate_open" is false you must not reveal ANY claim information: no claim IDs, statuses, denial reasons, amounts,
  dates, documents or deadlines, and do not confirm or deny that any claim exists. You may refer back to what the
  caller themselves said ("your claim from January") and promise to look once verification is complete.
- Never say the caller is verified unless the plan contains a "verified" or "consent_approved" act.
- Never offer actions that are not in the plan (no approving, overturning, paying, waiving or changing claims).
- An "empathize" act comes first: acknowledge the feeling sincerely and briefly (one sentence), without groveling,
  then continue with the rest of the plan. With an "explain_verification" act, explain why the step protects the caller.
- "decline_out_of_scope": politely say you can only help with insurance claims and this call, without answering
  the off-topic question at all, then steer back.
- Use the caller's name sparingly: in the first reply after verification and in the goodbye, rarely otherwise.
- "answer": answer every part of the caller's actual question first, in plain words, even if you covered it in an
  earlier turn (they are asking again for a reason). Then add at most two other points that matter now (e.g. a
  deadline). Don't pad the reply with unrequested guidance you already gave.
- "ask_identity": say "any three of" the listed details (one list, joined with "or"), and how many are still needed.
- "ask_intent" with already_listed=true: the claims were already shown; ask which one without repeating the list.
  With caller_asked_for_list=true the caller explicitly asked to see their claims: always list every claim again.
- "decline_out_of_scope" with medical=true: kindly say you can't interpret medical results and that their doctor or
  the provider who ordered the test is the right person; offer to keep helping with the claim.
- Ask at most one question at the end of the reply. Keep it concise: usually 2-6 sentences; use a short bulleted
  list only for documents or several claims. No headings, no markdown tables, no emoji, no sign-off names.
- Write only the reply text.

What some acts require:
- remember_for_later: say explicitly that you've noted what they're calling about (in their own terms, from "hint")
  and will pull it up right after verification.
- ask_identity: say how many more details are needed and list the allowed options.
- offer_email: ASK whether they want a summary of today's conversation (what was discussed, claim status, next steps)
  emailed to the masked address, and make clear they can skip it. Do not say you are sending it.
- send_email: confirm the summary was sent to the masked address.
- offer_human: ask whether they'd like to be transferred to a human representative.
- handoff: state that you ARE transferring them to a human representative now (not a question), and that the notes
  go with them so they won't have to start over. Do not ask any question.
- consent_requested: say the details check out, and that because they are calling on the policyholder's behalf,
  a consent request has been sent to the policyholder's phone ending in the given digits; once it's approved you can
  discuss the claim. Ask them to let you know when it's approved.
- consent_pending / consent_timeout: the request is still not approved / was not approved in time.
- email_on_file_only: explain the summary can only go to the masked email on file, and ask: send there, or skip?
- no_claims_on_file: state plainly that there are no claims on file under this policy (if "hint" is given, say you
  don't see that claim because there are none on file). Mention that a very recently submitted claim may not show yet.
  Do NOT ask for a claim number or date, and do NOT say you cannot see or list their claims.
- ask_intent / disambiguate / case_not_found / no_matching_case: show every claim in the list given, as bullets.
- verify_failed: say the details don't match the records (never say which), and ask them to double-check and correct
  any detail or give different ones (any three in total). Mention attempts_left.
- answer: if facts.appeal_deadline_status.passed is true, say clearly that the appeal deadline (give the date) has passed;
  do NOT tell them to submit "within a week" or imply the appeal is still open; explain that a human claims
  representative has to review whether late documents can still be considered. If it has not passed, you may mention
  how many days are left.
- unsupported_request: say plainly you can't do that here ("new_claim": filing a new claim; "account_change": changing
  account or contact details; "coverage_question": explaining policy coverage or benefits). Do not use an existing claim
  as a stand-in and do not invent procedures.
- crisis_support: the caller may be in crisis. Respond with genuine care first; give the 988 Suicide & Crisis Lifeline
  (call or text 988, in the US) and 911 for immediate danger. Do not mention claims, documents or deadlines in this reply.
- pii_caution: briefly and kindly say you only need the last 4 digits of the SSN, so they shouldn't share the full
  number in chat; it has been removed from the transcript.
- ask_rep_name: thank them; since they're calling on the policyholder's behalf, ask for their own full name first.
- offer_email with requested=true: they asked for it, so confirm you can send it to the masked address and ask
  whether to send it now (they can still skip it).
- answer with second_claim=true: they asked about two claims; answer this second claim too, from its own facts.
- consent_declined: the policyholder declined the request, so you can't discuss their claims with this caller.
- goodbye: a brief warm closing addressed to "first_name" (the person you are talking to).
- Names: when you address the caller by name, use exactly "address_caller_as" (if given). Never shorten, reorder
  or guess a first name from a full name."""


def _d(iso):
    return date.fromisoformat(iso).strftime("%B %-d, %Y")


def _join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _money(s):
    return f"${float(s):,.2f}"


EMPATHY = {
    "frustrated": "I hear you, and I'm sorry this has been frustrating.",
    "angry": "I completely understand why you're upset, and I'm sorry for the hassle.",
    "anxious": "I understand this is stressful. Let's work through it together.",
    "confused": "No problem, let me make this clearer.",
    "sad": "I'm sorry you're dealing with this.",
}


def _answer_text(a):
    f = a["facts"]
    c = f["claim"]
    intent = a.get("intent")
    out = [f"Your {c['case_type']} claim {c['case_id']}, filed {_d(c['created_at'])}, is currently {c['status']}."]
    docs = c.get("documents_needed", [])
    if c["status"] == "denied":
        out.append(f"It was denied because {c['denial_reason']}.")
        if docs:
            out.append("To have it reconsidered, please send: " + ", ".join(docs) + ".")
        ds = f.get("appeal_deadline_status")
        if ds and ds["passed"]:
            out.append(f"The appeal deadline was {_d(c['appeal_deadline'])}, and it has now passed, so a human claims "
                       "representative would need to review whether late documents can still be considered.")
        elif c.get("appeal_deadline"):
            out.append(f"The appeal deadline is {_d(c['appeal_deadline'])}.")
    elif c["status"] == "open":
        out.append(f"It's still being processed. The expected reimbursement is {_money(c['expected_reimbursement_amount'])} "
                   f"(the allowed maximum is {_money(c['allowed_max_amount'])}), and no payment has been issued yet.")
    elif c["status"] == "closed":
        out.append(f"It was settled with a net payment of {_money(c['net_pay'])} against an allowed amount of "
                   f"{_money(c['allowed_max_amount'])}.")
    by_topic = {g["topic"]: g["text"] for g in f["followup_guidance"]}
    topics = list(f["matched_followup_topics"])
    if not topics and docs:
        topics = {"document_submission": ["submission_method", "file_format_requirements"],
                  "next_steps": ["submission_timing", "processing_time_after_submission"],
                  "denial_question": ["submission_timing"],
                  "appeal": ["submission_timing", "submission_method"]}.get(intent, [])
    for t in topics:
        if t in by_topic:
            out.append(by_topic[t])
    if intent in ("document_submission",) and docs:
        for d in f["documents"]:
            if d["how_to_prepare"]:
                out.append(f"For the {d['document']}: {d['how_to_prepare']}")
    if intent == "payment_question" and c["status"] != "open":
        out.append(f"The expected reimbursement on file is {_money(c['expected_reimbursement_amount'])}.")
    q = (a.get("question") or "").lower()
    if docs and re.search(r"don'?t have|can'?t get|cannot get|can'?t find|lost|unavailable|not available", q):
        out = out[:1] + [d["if_unavailable"] for d in f["documents"]]
    return " ".join(out)


def render_template(plan, memory) -> str:
    parts = []
    for a in plan.acts:
        t = a["type"]
        if t == "empathize":
            parts.append(EMPATHY.get(a["emotion"], ""))
        elif t == "closed":
            parts.append("This conversation has been transferred to a human representative, who will pick it up shortly."
                         if a["phase"] == "HANDOFF" else
                         "This conversation has ended. Please start a new chat if you need anything else.")
        elif t == "decline_out_of_scope":
            parts.append("I'm not able to interpret medical results. That's a question for your doctor or the provider "
                         "who ordered the test." if a.get("medical") else
                         "I'm sorry, but I can only help with insurance claims and questions about your policy, "
                         "so I can't help with that one.")
        elif t == "offer_human":
            parts.append("Would you like me to transfer you to a human representative?")
        elif t == "remember_for_later":
            s = ""
            if a.get("hint"):
                s = f"I've noted that you're calling about your {a['hint']}, and I'll pull it up as soon as you're verified."
            if a.get("wants_details"):
                s += " I can't share any claim details until then."
            parts.append(s.strip())
        elif t == "explain_verification":
            parts.append("Claim records contain protected medical and financial information, so I have to confirm "
                         "I'm speaking with the policyholder before sharing anything. It's what keeps someone else "
                         "from accessing your claim.")
        elif t == "ask_identity":
            s = ""
            if a["rep_name_missing"]:
                s += "Could you also tell me your own full name? "
            who = "the policyholder's" if a["representative"] else "your"
            if a["have"]:
                s += (f"So far I have {who} {_join(a['have'])}. I need {a['need']} more of the following: "
                      f"{', '.join(a['options'])}.") if a["need"] else ""
            else:
                s += (f"To protect {who.replace('the ', 'the ')} account, please verify identity with any three of: "
                      "full name, date of birth, phone number, email address, or the last 4 digits of the SSN or national ID.")
            parts.append(s)
        elif t == "verify_failed":
            parts.append("I'm sorry, but those details don't match what we have on file, and for security I can't say "
                         "which one. Could you double-check and correct them, or give any three of: " + ", ".join(a["options"])
                         + f"? ({a['attempts_left']} attempt{'s' if a['attempts_left'] != 1 else ''} left before I connect you with a representative.)")
        elif t == "verified":
            parts.append(f"Thank you, {a['first_name']}, you're verified.")
        elif t == "rep_not_authorized":
            parts.append(f"Thank you. I'm not able to find you listed as an authorized representative on "
                         f"{a['holder_first']}'s policy, so I can't share claim details on this call.")
        elif t == "consent_requested":
            parts.append(f"Thank you, those details check out. Because you're calling on {a['holder_first']}'s behalf, "
                         f"I've sent a consent request to the phone number on file ending in {a['phone_last2']}. "
                         "Once it's approved I can go over the claim with you. Just let me know when it's done.")
        elif t == "consent_pending":
            parts.append("The consent request is still pending on our side. We need the policyholder's approval before "
                         "I can discuss claim details. Let me know once it's been approved.")
        elif t == "consent_approved":
            parts.append(f"{a['holder_first']} has approved access. Thank you for waiting.")
        elif t == "consent_declined":
            parts.append("The policyholder has declined the request, so I'm not able to discuss their claims with you.")
        elif t == "consent_timeout":
            parts.append("I still haven't received the policyholder's consent, so I'm not able to share claim details.")
        elif t == "case_selected":
            lead = "Based on what you mentioned earlier, I found" if a["from_memory"] else "I found"
            parts.append(f"{lead} your {a['label']}.")
        elif t == "disambiguate":
            parts.append("I see more than one claim that could match. Which one do you mean?\n"
                         + "\n".join(f"- {o}" for o in a["options"]))
        elif t == "crisis_support":
            parts.append("I'm really sorry you're feeling this way, and I'm glad you said something. You don't have to go "
                         "through this alone. If you're in the US, you can call or text 988 to reach the Suicide & Crisis "
                         "Lifeline any time, and if you're in immediate danger, please call 911.")
        elif t == "pii_caution":
            parts.append("Just so you know, I only need the last 4 digits of your SSN, so please don't share the full "
                         "number in chat. I've removed it from our conversation.")
        elif t == "ask_rep_name":
            parts.append(f"Thank you. Since you're calling on {a['holder_first']}'s behalf, could you tell me your own full name?")
        elif t == "unsupported_request":
            parts.append({"new_claim": "I'm not able to file a new claim through this chat.",
                          "account_change": "I'm not able to change account or contact details through this chat.",
                          "coverage_question": "I don't have your policy's coverage details here, so I can't say what's covered."
                          }.get(a["kind"], "That's not something I can help with in this chat."))
        elif t == "no_claims_on_file":
            parts.append(("I've checked your policy and I don't see " + (f"a {a['hint']}" if a.get("hint") else "any claims")
                          + " on file. In fact, there are no claims on file under your policy right now."
                          if a.get("hint") else
                          "I've checked your policy, and there are no claims on file under your account right now.")
                         + " If you submitted a claim very recently, it may not be in the system yet.")
        elif t == "case_not_found":
            parts.append(f"I don't see claim {a['case_id']} on your policy. Here's what I do see:\n"
                         + "\n".join(f"- {o}" for o in a["claims"]) + "\nWhich one would you like to discuss?")
        elif t == "no_matching_case":
            parts.append(f"I couldn't find a {a['hint']} on your policy. Here's what I do see:\n"
                         + "\n".join(f"- {o}" for o in a["claims"]) + "\nWhich one would you like to discuss?")
        elif t == "ask_intent":
            parts.append("Which of the claims I listed would you like to go over?" if a.get("already_listed") else
                         "Which claim can I help you with today? I see:\n" + "\n".join(f"- {o}" for o in a["claims"]))
        elif t == "answer":
            parts.append(_answer_text(a))
        elif t == "ask_case_need":
            parts.append("What would you like to know about this claim?")
        elif t == "ask_anything_else":
            parts.append("Is there anything else I can help you with?")
        elif t == "action_not_allowed":
            parts.append("I'm not able to change claim decisions, amounts or payments from here. What I can do is explain "
                         "the decision and the steps to have the claim reconsidered.")
        elif t == "offer_email":
            parts.append(("Would you like" if a["repeat"] else "Before we wrap up, would you like")
                         + f" me to email a summary of today's conversation, including the claim status and next steps, "
                         f"to {a['email']}? Or we can skip it.")
        elif t == "send_email":
            parts.append(f"Done. I've emailed the summary to {a['masked']}.")
        elif t == "email_skipped":
            parts.append("No problem, I won't send an email.")
        elif t == "email_on_file_only":
            parts.append(f"For security, I can only send the summary to the email on file ({a['masked']}). "
                         "Would you like me to send it there, or skip it?")
        elif t == "goodbye":
            parts.append(f"Thanks for reaching out, {a['first_name']}. Take care!")
        elif t == "handoff":
            s = "I'm transferring you to a human representative now. I've passed along a note so you won't have to start over."
            if not a["verified"]:
                s += " They'll also need to confirm your identity before discussing claim details."
            parts.append(s)
    return "\n\n".join(p for p in parts if p).replace("\n\n- ", "\n- ")


# ------------------------------------------------------------------ guard
CLAIM_ID = re.compile(r"\bCL-\d+\b", re.I)
MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
MONTH_DAY = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\b")
ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
VERIFIED_CLAIM = re.compile(r"\b(you'?re|you are|you have been|you've been)\s+(now\s+)?(successfully\s+)?verified\b", re.I)


def _protected_terms():
    terms = set()
    for c in BACKEND.claims:
        terms.update(d.lower() for d in c.get("documents_needed", []))
        terms.add(c["case_id"].lower())
    return terms | {"appeal deadline", "denial reason"}


PROTECTED = _protected_terms()


def guard(reply: str, plan, memory) -> list[str]:
    """Return the list of violations in a drafted reply (empty = safe)."""
    v = []
    lo = reply.lower()
    plan_json = json.dumps(plan.acts).lower()
    if not memory.gate_open:
        for term in PROTECTED:
            if term in lo:
                v.append(f"claim detail '{term}' disclosed before the gate opened")
        if MONEY.search(reply):
            v.append("amount disclosed before the gate opened")
        if VERIFIED_CLAIM.search(reply) and not any(a["type"] in ("verified",) for a in plan.acts):
            v.append("claimed verification that did not happen")
        return v + _act_checks(reply, plan)
    # Facts already established in this conversation (selected / discussed claims) also count as grounded.
    known = [BACKEND.claim(memory.verified_party_id, cid) for cid in {memory.selected_case_id, *memory.cases_discussed} if cid]
    plan_json += json.dumps([c for c in known if c]).lower()
    own_ids = {c["case_id"].lower() for c in BACKEND.claims_for(memory.verified_party_id)}
    echoed = {a["case_id"].lower() for a in plan.acts if a["type"] == "case_not_found"}
    for cid in CLAIM_ID.findall(reply):
        if cid.lower() in echoed:
            continue
        if cid.lower() not in own_ids or cid.lower() not in plan_json:
            v.append(f"claim id {cid} not grounded in this turn")
    allowed_amounts = {float(x) for x in re.findall(r'"(\d+\.\d{2})"', plan_json)}
    for amt in MONEY.findall(reply):
        if float(amt.replace(",", "")) not in allowed_amounts:
            v.append(f"amount ${amt} not grounded")
    allowed_dates = set(ISO.findall(plan_json))
    allowed_md = {(int(d[5:7]), int(d[8:10])) for d in allowed_dates}
    months = ["january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december"]
    # claim labels carry dates already written out ("january 12, 2026")
    allowed_md |= {(months.index(mon) + 1, int(day)) for mon, day in re.findall(
        r"\b(" + "|".join(months) + r")\s+(\d{1,2})\b", plan_json)}
    for mon, day in MONTH_DAY.findall(reply):
        if (months.index(mon.lower()) + 1, int(day)) not in allowed_md:
            v.append(f"date {mon} {day} not grounded")
    for d in ISO.findall(reply):
        if d not in allowed_dates:
            v.append(f"date {d} not grounded")
    for mon, day in re.findall(r"(\d{1,2})月(\d{1,2})日", reply):
        if (int(mon), int(day)) not in allowed_md:
            v.append(f"date {mon}月{day}日 not grounded")
    return v + _act_checks(reply, plan)


def _act_checks(reply, plan):
    """Required-shape checks: an offered choice must be asked, a transfer must be stated."""
    v = []
    types = [a["type"] for a in plan.acts]
    requested = any(a["type"] == "offer_email" and a.get("requested") for a in plan.acts)
    if "offer_email" in types and ("?" not in reply or not (requested or re.search(r"skip|no need|not necessary|rather not|or not", reply, re.I))):
        v.append("email offer did not ask the caller to choose send or skip")
    if "crisis_support" in types and "988" not in reply:
        v.append("crisis reply must include the 988 lifeline")
    if "handoff" in types and (reply.rstrip().endswith("?") or not re.search(r"transfer|connect", reply, re.I)):
        v.append("handoff must state the transfer, not ask")
    return v


def llm_render(llm, plan, memory, history, utterance):
    payload = {
        "phase_before": plan.phase_before,
        "phase_now": memory.phase,
        "gate_open": memory.gate_open,
        "caller_emotion": plan.emotion,
        "address_caller_as": memory.address_as,
        "plan": plan.acts,
        "recent_conversation": history[-8:],
        "caller_latest_message": utterance,
    }
    return llm.text(SYSTEM, json.dumps(payload, default=str))


# ------------------------------------------------------------------ email summary
SUMMARY_SYSTEM = """You write the follow-up email an insurance claims support agent sends after a call.
Use ONLY the facts provided. Structure: a one-line greeting; "What we discussed" (short bullets); "Claim status"
(one bullet per claim with its status/outcome); "Next steps" (bullets with documents, deadlines and timing if present).
Plain text, no markdown headers (use simple capitalized section labels), under 220 words, signed "Claims Support Team".
Never add amounts, dates, documents or promises that are not in the facts."""


def summary_facts(memory):
    ph = BACKEND.policyholder(memory.verified_party_id)
    claims = [BACKEND.claim(memory.verified_party_id, cid) for cid in memory.cases_discussed]
    nexts = []
    for c in claims:
        if c.get("documents_needed"):
            nexts.append(f"Submit for claim {c['case_id']}: {', '.join(c['documents_needed'])} (within a week; "
                         "use the member portal or claim upload link).")
            if c.get("appeal_deadline"):
                nexts.append(f"Appeal deadline for claim {c['case_id']}: {_d(c['appeal_deadline'])}.")
            nexts.append("Expected review time once the documents are received: "
                         + BACKEND.guideline["claim_followup_settings"]["average_processing_time_after_submission"]["en"] + ".")
    return {
        "recipient_name": ph["name"],
        "caller_role": memory.caller_role,
        "representative_name": memory.rep_name,
        "questions_discussed": [d["question"] for d in memory.discussed],
        "claims": claims,
        "claims_on_file": len(BACKEND.claims_for(memory.verified_party_id)),
        "next_steps": nexts,
    }


def template_summary(f):
    lines = [f"Hello {f['recipient_name']},", "", "Thank you for contacting Claims Support. Here is a summary of our conversation.", "",
             "WHAT WE DISCUSSED"]
    lines += [f"- {q}" for q in f["questions_discussed"]] or ["- General questions about your claims"]
    lines += ["", "CLAIM STATUS"]
    if not f["claims"]:
        lines.append("- No claims were found on file under your policy.")
    for c in f["claims"]:
        s = f"- {c['case_id']} ({c['case_type']}, filed {_d(c['created_at'])}): {c['status']}"
        if c["status"] == "denied":
            s += f" because {c['denial_reason']}"
        elif c["status"] == "closed":
            s += f", net payment {_money(c['net_pay'])}"
        lines.append(s + ".")
    if f["next_steps"]:
        lines += ["", "NEXT STEPS"] + [f"- {n}" for n in f["next_steps"]]
    lines += ["", "If you have questions, reply to this email or start a new chat.", "", "Claims Support Team"]
    return "\n".join(lines)


def compose_summary(llm, memory):
    f = summary_facts(memory)
    if llm is not None:
        text = llm.text(SUMMARY_SYSTEM, json.dumps(f, default=str))
        if text:
            amounts = {float(x) for x in re.findall(r'"(\d+\.\d{2})"', json.dumps(f))}
            if all(float(a.replace(",", "")) in amounts for a in MONEY.findall(text)):
                return text, "llm"
    return template_summary(f), "template"
