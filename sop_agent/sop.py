"""Deterministic SOP engine.

The engine owns phase order, safety gates and the set of allowed actions.
Each turn it receives the caller's structured extraction, updates memory and
returns a *plan*: an ordered list of acts the reply must perform, plus the
only facts the reply may use. The LLM phrases the plan; it cannot change it.

    VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS -> ENDED
         \\____________________ HANDOFF (human) ____________________/
"""
from __future__ import annotations

import os
import re
from datetime import date

from .data import BACKEND, FIELD_LABELS, ID_FIELDS, mask_email
from .memory import Memory

NEGATIVE = {"frustrated", "angry", "anxious", "confused", "sad"}
MAX_VERIFY_FAILURES = 3
MAX_OFF_TOPIC = 3            # 3rd strike -> offer human, 4th -> transfer
MAX_REFUSALS = 3             # persuasion limit in VERIFY_ID
DISALLOWED_ACTION = re.compile(
    r"\b(approve|overturn|reverse|waive|override|pay me|change (the|my) (status|amount)|mark it|"
    r"cancel (my|the) policy|refund|reopen)\b", re.I)


class Plan:
    def __init__(self, phase_before):
        self.phase_before = phase_before
        self.acts: list[dict] = []
        self.emotion = "neutral"
        self.trace: list[str] = []

    def add(self, type_, **data):
        self.acts.append({"type": type_, **data})

    def has(self, type_):
        return any(a["type"] == type_ for a in self.acts)


def _hint_text(h):
    parts = []
    if h.get("status"):
        parts.append(h["status"])
    if h.get("case_type"):
        parts.append(h["case_type"])
    s = " ".join(parts + ["claim"])
    if h.get("month"):
        s += " from " + date(2000, int(h["month"]), 1).strftime("%B")
        if h.get("year"):
            s += f" {h['year']}"
    if h.get("case_id"):
        s = f"claim {h['case_id']}"
    return s


UNSUPPORTED = {"new_claim", "account_change", "coverage_question"}
LOCKOUT_MAX = int(os.environ.get("LOCKOUT_MAX_FAILURES", "5"))          # across all chats
LOCKOUT_WINDOW_S = float(os.environ.get("LOCKOUT_WINDOW_MIN", "30")) * 60


class Engine:
    def __init__(self, backend=BACKEND, today: str | None = None, store=None, session_id: str | None = None):
        self.b = backend
        self.today = today or date.today().isoformat()
        self.store = store          # lockout + consent persistence (None = per-chat limits only)
        self.session_id = session_id

    def _consent_service(self, m):
        from .integrations import consent_service
        return consent_service(self.b, self.store, m.consent_scenario)

    # ------------------------------------------------------------------ entry
    def step(self, m: Memory, x: dict, utterance: str) -> Plan:
        p = Plan(m.phase)
        p.emotion = x.get("emotion", "neutral")
        new_info = self._remember(m, x, p)

        m.emotions.append(p.emotion)
        m.negative_streak = m.negative_streak + 1 if p.emotion in ("frustrated", "angry") else 0
        # Off-topic questions are not distress; only acknowledge real frustration there.
        if p.emotion in NEGATIVE and not (x.get("scope") == "out_of_scope" and p.emotion in ("confused", "anxious")):
            p.add("empathize", emotion=p.emotion)

        if m.phase in ("ENDED", "HANDOFF"):
            p.add("closed", phase=m.phase)
            return p

        # Safety first: a caller in crisis gets support, not the workflow.
        if x.get("safety_concern"):
            p.acts = []
            p.add("crisis_support")
            self._offer_human(m, p, "caller expressed thoughts of self-harm")
            p.trace.append("SAFETY: crisis support, workflow paused")
            return p
        if x.get("full_ssn_shared"):
            p.add("pii_caution")
            p.trace.append("full SSN redacted from the transcript")

        # A request for a human is always honoured.
        if x.get("wants_human") or (m.last_agent_question == "offer_human" and x.get("confirmation") == "yes"):
            return self._handoff(m, p, "caller asked for a human representative")
        if m.last_agent_question == "offer_human" and x.get("confirmation") == "no":
            m.human_offered = False
            if not x.get("refuses_to_verify"):
                m.refusals, m.negative_streak = 0, 0
            p.trace.append("caller declined human transfer; continuing SOP")

        # Scope guard.
        if x.get("medical"):
            p.trace.append("medical interpretation question -> refer to the treating provider")
            p.add("decline_out_of_scope", strike=0, medical=True)
            self._reprompt(m, p)
            return p
        if x.get("scope") == "out_of_scope" and not new_info:
            m.off_topic_streak += 1
            p.trace.append(f"out-of-scope strike {m.off_topic_streak}")
            if m.off_topic_streak > MAX_OFF_TOPIC:
                return self._handoff(m, p, "caller kept asking out-of-scope questions")
            p.add("decline_out_of_scope", strike=m.off_topic_streak)
            if m.off_topic_streak == MAX_OFF_TOPIC:
                self._offer_human(m, p, "repeated out-of-scope questions")
                return p
            self._reprompt(m, p)
            return p
        m.off_topic_streak = 0

        if m.gate_open and m.intent in UNSUPPORTED and m.phase in ("RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"):
            return self._unsupported(m, p)

        handlers = {"VERIFY_ID": self._verify, "RESOLVE_INTENT": self._resolve,
                    "PROCESS_CASE": self._process, "POST_PROCESS": self._post}
        # Handlers may advance the phase and chain into the next one in the same turn.
        for _ in range(4):
            phase = m.phase
            if m.gate_open and m.intent in UNSUPPORTED and phase in ("RESOLVE_INTENT", "PROCESS_CASE"):
                self._unsupported(m, p)   # e.g. intent remembered during VERIFY_ID
                break
            if m.gate_open and m.intent == "email_summary" and phase in ("RESOLVE_INTENT", "PROCESS_CASE"):
                m.intent, m.pending_question, m.email_requested = None, None, True
                m.email_state = "not_offered"
                m.phase = phase = "POST_PROCESS"
                p.trace.append("caller asked for the email summary -> POST_PROCESS")
            handlers[phase](m, x, utterance, p)
            if m.phase == phase or m.phase not in handlers:
                break
            p.trace.append(f"phase {phase} -> {m.phase}")
        return p

    def _unsupported(self, m: Memory, p: Plan):
        """Requests outside this agent's allowed actions: say so, never repurpose an existing claim."""
        kind = m.intent
        p.trace.append(f"unsupported request: {kind}")
        p.add("unsupported_request", kind=kind)
        m.intent, m.pending_question = None, None
        if not m.human_offered:
            self._offer_human(m, p, f"{kind} needs a human representative")
        else:
            p.add("ask_anything_else")
            m.last_agent_question = "anything_else"
        return p

    # ------------------------------------------------------------- memory
    def _remember(self, m: Memory, x: dict, p: Plan) -> bool:
        """Store anything useful, whatever phase it belongs to. Returns True
        when the utterance carried new SOP-relevant information."""
        new = False
        if not m.verified_party_id:
            for k, v in (x.get("identity") or {}).items():
                if v and m.identity.get(k) != v:
                    m.identity[k] = v
                    new = True
                    p.trace.append(f"memory: identity.{k} captured")
        if x.get("preferred_name"):
            m.preferred_name = x["preferred_name"].strip()
            if m.address_as:
                m.address_as = m.preferred_name
            p.trace.append(f"memory: preferred name '{m.preferred_name}'")
        if x.get("caller_role") in ("policyholder", "representative") and not m.verified_party_id:
            m.caller_role = x["caller_role"]
        if x.get("rep_name"):
            m.rep_name = x["rep_name"]
            new = True
        elif m.last_agent_question == "rep_name" and (x.get("identity") or {}).get("full_name"):
            m.rep_name = x["identity"]["full_name"]   # answer to "what's your own name?"
            new = True
        if x.get("rep_relationship"):
            m.rep_relationship = x["rep_relationship"]
        hints = {k: v for k, v in (x.get("case_hints") or {}).items() if v}
        if hints:
            if hints.get("case_id"):
                hints["case_id"] = hints["case_id"].upper()
            m.case_hints.update(hints)
            new = True
            p.trace.append(f"memory: case hints {hints} (phase {m.phase})")
        if x.get("intent") and x["intent"] != "none":
            m.intent = x["intent"]
            new = True
            p.trace.append(f"memory: intent={m.intent}")
        if x.get("question") and x.get("scope") != "out_of_scope":
            m.pending_question = x["question"]
        p._turn_hints = hints  # this turn only, used for disambiguation
        return new

    # ------------------------------------------------------------- helpers
    def _handoff(self, m: Memory, p: Plan, reason: str):
        m.phase, m.handoff_reason = "HANDOFF", reason
        note = {
            "reason": reason,
            "identity_verified": bool(m.verified_party_id),
            "party_id": m.verified_party_id,
            "caller_role": m.caller_role,
            "intent": m.intent,
            "case_hints": m.case_hints,
            "selected_case": m.selected_case_id if m.gate_open else None,
            "recent_emotions": m.emotions[-5:],
        }
        m.handoff_note = note
        p.add("handoff", reason=reason, verified=bool(m.verified_party_id))
        p.trace.append(f"HANDOFF: {reason}")
        return p

    def _offer_human(self, m: Memory, p: Plan, why: str):
        m.human_offered = True
        m.last_agent_question = "offer_human"
        p.add("offer_human", why=why)

    def _reprompt(self, m: Memory, p: Plan):
        """After a detour, steer back to what the current phase needs."""
        if m.phase == "VERIFY_ID":
            self._ask_identity(m, p)
        elif m.phase == "RESOLVE_INTENT":
            p.add("ask_intent", claims=[self.b.claim_label(c) for c in self.b.claims_for(m.verified_party_id)],
                  already_listed=m.claims_listed)
            m.claims_listed = True
        elif m.phase == "PROCESS_CASE":
            p.add("ask_anything_else")
        elif m.phase == "POST_PROCESS" and m.email_state == "offered":
            p.add("offer_email", email=mask_email(self.b.policyholder(m.verified_party_id)["email"]), repeat=True)

    def _ask_identity(self, m: Memory, p: Plan):
        have = [f for f in ID_FIELDS if m.identity.get(f)]
        missing = [f for f in ID_FIELDS if f not in have]
        p.add("ask_identity", have=[FIELD_LABELS[f] for f in have], need=max(0, 3 - len(have)),
              options=[FIELD_LABELS[f] for f in missing],
              representative=m.caller_role == "representative", rep_name_missing=
              m.caller_role == "representative" and not m.rep_name)
        m.last_agent_question = "identity"

    # ------------------------------------------------------------- VERIFY_ID
    def _verify(self, m: Memory, x: dict, utt: str, p: Plan):
        if m.verified_party_id:            # representative waiting on consent
            return self._consent(m, x, p)

        if x.get("refuses_to_verify"):
            m.refusals += 1
            p.trace.append(f"verification pushback #{m.refusals}")

        provided = {k: v for k, v in m.identity.items() if v and k in (*ID_FIELDS, "policy_number")}
        n_core = sum(1 for k in provided if k in ID_FIELDS)

        if n_core >= 3:
            pid, matched, mismatched = self.b.match_identity(provided)
            core = [f for f in matched if f in ID_FIELDS]
            if self.store and pid and core and self.store.recent_failures(pid, LOCKOUT_WINDOW_S) >= LOCKOUT_MAX:
                # Too many failures for this policyholder across chats: stop, even if this attempt is right.
                p.trace.append(f"LOCKOUT: {pid} has >= {LOCKOUT_MAX} failed attempts in the window")
                return self._handoff(m, p, "verification temporarily locked after repeated failed attempts")
            if len(core) >= 3 and not mismatched:
                m.verified_party_id, m.verified_fields = pid, matched
                p.trace.append(f"VERIFIED party {pid} on {matched}")
                if self.store:
                    self.store.clear_failures(pid)
                if m.caller_role == "representative":
                    m.address_as = m.preferred_name or m.rep_name
                    return self._start_consent(m, p)
                m.phase = "RESOLVE_INTENT"
                ph = self.b.policyholder(pid)
                m.address_as = m.preferred_name or self.b.address_name(ph)
                p.add("verified", first_name=m.address_as)
                return
            if self.store and pid and core:
                self.store.record_failure(pid)
            m.verify_failures += 1
            p.trace.append(f"verification failed (attempt {m.verify_failures}); matched={len(core)} mismatched={len(mismatched)}")
            m.last_failed_identity = dict(provided)  # kept, so the caller can correct just one detail
            if m.verify_failures >= MAX_VERIFY_FAILURES:
                return self._handoff(m, p, "identity could not be verified after 3 attempts")
            p.add("verify_failed", attempts_left=MAX_VERIFY_FAILURES - m.verify_failures,
                  options=[FIELD_LABELS[f] for f in ID_FIELDS])
            m.last_agent_question = "identity"
            return

        # Not enough identity yet: keep the gate closed, keep the call moving.
        hints = getattr(p, "_turn_hints", {})
        if hints or (x.get("intent") not in (None, "none")):
            # never echo a claim number before verification, even the caller's own
            h = {k: v for k, v in m.case_hints.items() if k != "case_id"}
            p.add("remember_for_later",
                  hint=_hint_text(h) if h else ("the claim number you mentioned" if m.case_hints.get("case_id") else None),
                  wants_details=x.get("intent") not in (None, "none"))
        if x.get("refuses_to_verify") or (p.emotion in ("frustrated", "angry") and (x.get("question") or hints or x.get("intent") not in (None, "none"))):
            p.add("explain_verification")
        if m.refusals >= MAX_REFUSALS:
            return self._handoff(m, p, "caller declined identity verification repeatedly")
        self._ask_identity(m, p)
        if (m.refusals >= 2 or m.negative_streak >= 3) and not m.human_offered:
            self._offer_human(m, p, "verification is stuck")

    def _start_consent(self, m: Memory, p: Plan):
        ph = self.b.policyholder(m.verified_party_id)
        holder = self.b.address_name(ph)
        if not m.rep_name:
            p.add("ask_rep_name", holder_first=holder)
            m.last_agent_question = "rep_name"
            return
        m.address_as = m.preferred_name or m.rep_name
        rep = self.b.representative_for(m.rep_name or "", m.verified_party_id)
        if not rep:
            p.trace.append("representative not on file for this policy")
            p.add("rep_not_authorized", holder_first=holder)
            self._offer_human(m, p, "representative is not authorized on the policy")
            return
        svc = self._consent_service(m)
        req = svc.request(m, ph, self.session_id)
        m.consent_status, m.consent_ref, m.consent_link = req["status"], req["ref"], req["link"]
        p.trace.append(f"tool: consent request via {svc.name}" + (f" -> {req['link']}" if req["link"] else ""))
        p.add("consent_requested", holder_first=holder, phone_last2=ph["phone"][-2:])

    def _consent(self, m: Memory, x: dict, p: Plan):
        if m.consent_status is None and not m.human_offered:
            return self._start_consent(m, p)   # still waiting for the representative's own name
        if m.consent_status != "pending":
            self._offer_human(m, p, "consent unavailable")
            return
        status = self._consent_service(m).status(m)
        p.trace.append(f"consent poll #{m.consent_polls}: {status}")
        if status == "approved":
            m.consent_status = "approved"
            m.phase = "RESOLVE_INTENT"
            p.add("consent_approved", holder_first=self.b.address_name(self.b.policyholder(m.verified_party_id)))
        elif status == "declined":
            m.consent_status = "declined"
            p.add("consent_declined")
            self._offer_human(m, p, "policyholder declined consent")
        elif status == "timeout":
            m.consent_status = "timeout"
            p.add("consent_timeout")
            self._offer_human(m, p, "consent not received")
        else:
            p.add("consent_pending", explain=x.get("emotion") in NEGATIVE)

    # ------------------------------------------------------------- RESOLVE_INTENT
    def _resolve(self, m: Memory, x: dict, utt: str, p: Plan):
        pid = m.verified_party_id
        all_claims = self.b.claims_for(pid)
        if not all_claims:
            return self._no_claims(m, x, p)
        if x.get("done") and not m.case_hints:
            m.phase = "POST_PROCESS"
            return
        if m.intent == "list_claims":
            p.add("ask_intent", claims=[self.b.claim_label(c) for c in all_claims], already_listed=False,
                  caller_asked_for_list=True)
            m.claims_listed = True
            m.intent, m.case_hints, m.candidate_case_ids = None, {}, []
            m.last_agent_question = "intent"
            return
        turn_hints = getattr(p, "_turn_hints", {})
        cands = []
        if m.candidate_case_ids and (turn_hints or x.get("question")):
            pool = [c for c in all_claims if c["case_id"] in m.candidate_case_ids]
            cands = self._filter(pool, turn_hints, utt)
        if not cands and m.case_hints:
            cands = self.b.find_claims(pid, m.case_hints)
        if not cands and turn_hints and turn_hints != m.case_hints:
            cands = self.b.find_claims(pid, turn_hints)
            if cands:
                m.case_hints = dict(turn_hints)

        if m.case_hints.get("case_id") and not cands:
            p.add("case_not_found", case_id=m.case_hints["case_id"],
                  claims=[self.b.claim_label(c) for c in all_claims])
            m.case_hints.pop("case_id")
            return
        if len(cands) == 1:
            c = cands[0]
            m.selected_case_id, m.candidate_case_ids = c["case_id"], []
            from_memory = p.phase_before == "VERIFY_ID"
            p.add("case_selected", label=self.b.claim_label(c), from_memory=from_memory,
                  hint=_hint_text(m.case_hints))
            p.trace.append(f"case resolved -> {c['case_id']} from hints {m.case_hints}"
                           + (" (remembered from VERIFY_ID)" if from_memory else ""))
            m.phase = "PROCESS_CASE"
            return
        if len(cands) > 1:
            m.candidate_case_ids = [c["case_id"] for c in cands]
            p.add("disambiguate", options=[self.b.claim_label(c) for c in cands])
            m.last_agent_question = "pick_case"
            return
        if m.case_hints:
            p.add("no_matching_case", hint=_hint_text(m.case_hints),
                  claims=[self.b.claim_label(c) for c in all_claims])
            m.case_hints = {}
            return
        p.add("ask_intent", claims=[self.b.claim_label(c) for c in all_claims], already_listed=m.claims_listed)
        m.claims_listed = True
        m.last_agent_question = "intent"

    def _no_claims(self, m: Memory, x: dict, p: Plan):
        """Verified caller with no claims on file: say so plainly (never ask for a
        claim number that cannot exist), offer a human, and wrap up if declined."""
        if x.get("done") or (m.last_agent_question == "offer_human" and x.get("confirmation") == "no"):
            p.trace.append("no claims on file; caller declined transfer -> wrap up")
            m.case_hints, m.intent = {}, None
            m.phase = "POST_PROCESS"
            return
        p.trace.append("verified party has no claims on file")
        p.add("no_claims_on_file", hint=_hint_text(m.case_hints) if m.case_hints else None)
        m.case_hints, m.intent, m.pending_question = {}, None, None
        self._offer_human(m, p, "caller expects a claim that is not on file")

    @staticmethod
    def _filter(pool, hints, utt):
        out = pool
        for k in ("case_id", "case_type", "status"):
            if hints.get(k):
                out = [c for c in out if str(c[k]).lower() == str(hints[k]).lower()]
        if hints.get("month"):
            out = [c for c in out if int(c["created_at"][5:7]) == int(hints["month"])]
        if hints.get("year"):
            out = [c for c in out if int(c["created_at"][:4]) == int(hints["year"])]
        if not hints:
            lo = utt.lower()
            for i, word in enumerate(("first", "second", "third", "fourth")):
                if word in lo and i < len(pool):
                    return [pool[i]]
            if "latest" in lo or "recent" in lo or "newest" in lo:
                return [max(pool, key=lambda c: c["created_at"])]
        return out if out != pool or len(pool) == 1 else []

    # ------------------------------------------------------------- PROCESS_CASE
    def _process(self, m: Memory, x: dict, utt: str, p: Plan):
        pid = m.verified_party_id
        turn_hints = getattr(p, "_turn_hints", {})
        current = self.b.claim(pid, m.selected_case_id)
        other_case = turn_hints and not p.has("case_selected") and any(
            turn_hints.get(k) and str(turn_hints[k]).lower() != str(current[k]).lower()
            for k in ("case_id", "case_type", "status"))
        if x.get("intent") == "list_claims" and not p.has("case_selected"):
            p.trace.append("caller asked to see all claims")
            m.selected_case_id, m.intent = None, "list_claims"
            m.phase = "RESOLVE_INTENT"
            return
        if (x.get("switch_case") and not p.has("case_selected")) or other_case:
            p.trace.append("caller switched to another claim")
            if m.selected_case_id not in m.cases_discussed and any(d["case_id"] == m.selected_case_id for d in m.discussed):
                m.cases_discussed.append(m.selected_case_id)
            m.selected_case_id = None
            m.case_hints = dict(turn_hints)
            m.phase = "RESOLVE_INTENT"
            return
        if x.get("done") or (m.last_agent_question == "anything_else" and x.get("confirmation") == "no"
                             and not x.get("question")):
            m.phase = "POST_PROCESS"
            return
        if DISALLOWED_ACTION.search(utt):
            p.add("action_not_allowed", request=utt)
            p.add("ask_anything_else")
            m.last_agent_question = "anything_else"
            return

        question = m.pending_question
        intent = m.intent
        if not question and (not intent or intent == "none") and not p.has("case_selected"):
            if x.get("scope") == "smalltalk" or x.get("confirmation") == "yes":
                p.add("ask_case_need")
                m.last_agent_question = "case_need"
                return
            question = utt
        if not question and (not intent or intent == "none"):
            p.add("ask_case_need")
            m.last_agent_question = "case_need"
            return
        question = question or {
            "denial_question": "Why was my claim denied and what is missing?",
            "claim_status": "What is the status of my claim?",
            "document_submission": "How do I submit the requested documents?",
            "next_steps": "What are my next steps?",
            "payment_question": "What was paid on this claim?",
            "appeal": "How do I appeal this decision?",
            "general_claim_question": "Can you tell me about this claim?",
        }.get(intent, "Can you tell me about this claim?")
        facts = self.b.case_facts(current, question, self.today)
        p.add("answer", question=question, intent=intent, facts=facts)
        for h in (x.get("additional_case_hints") or [])[:1]:
            h = {k: v for k, v in h.items() if v}
            other = self.b.find_claims(pid, h) if h else []
            if len(other) == 1 and other[0]["case_id"] != current["case_id"]:
                o = other[0]
                p.add("answer", question=question, intent=None, facts=self.b.case_facts(o, question, self.today),
                      second_claim=True)
                m.discussed.append({"case_id": o["case_id"], "question": question, "intent": None, "topics": []})
                if o["case_id"] not in m.cases_discussed:
                    m.cases_discussed.append(o["case_id"])
                p.trace.append(f"also answered for second claim {o['case_id']}")
        if facts.get("late_appeal_policy") and not m.human_offered:
            self._offer_human(m, p, "appeal deadline has passed")
        else:
            p.add("ask_anything_else")
        m.discussed.append({"case_id": current["case_id"], "question": question, "intent": intent,
                            "topics": facts["matched_followup_topics"]})
        if current["case_id"] not in m.cases_discussed:
            m.cases_discussed.append(current["case_id"])
        m.pending_question, m.intent = None, None
        if not p.has("offer_human"):
            m.last_agent_question = "anything_else"

    def _caller_first(self, m, ph):
        return m.address_as or self.b.address_name(ph)

    # ------------------------------------------------------------- POST_PROCESS
    def _post(self, m: Memory, x: dict, utt: str, p: Plan):
        ph = self.b.policyholder(m.verified_party_id)
        if m.email_state == "not_offered":
            m.email_state = "offered"
            m.last_agent_question = "email"
            p.add("offer_email", email=mask_email(ph["email"]), repeat=False, requested=m.email_requested)
            return
        new_email = (x.get("identity") or {}).get("email")
        if new_email and new_email.lower() not in {e.lower() for e in [ph["email"], *ph.get("email_aliases", [])]}:
            p.add("email_on_file_only", masked=mask_email(ph["email"]))
            return
        choice = x.get("email_choice")
        if choice is None and m.last_agent_question == "email" and x.get("confirmation"):
            choice = "send" if x["confirmation"] == "yes" else "skip"
        if choice == "send":
            m.email_state = "sent"
            p.add("send_email", to=ph["email"], masked=mask_email(ph["email"]))
            p.add("goodbye", first_name=self._caller_first(m, ph))
            m.phase = "ENDED"
            return
        if choice == "skip":
            m.email_state = "skipped"
            p.add("email_skipped")
            p.add("goodbye", first_name=self._caller_first(m, ph))
            m.phase = "ENDED"
            return
        if x.get("question") or (x.get("intent") not in (None, "none")) or getattr(p, "_turn_hints", {}):
            m.email_state = "not_offered"
            m.phase = "RESOLVE_INTENT" if not m.selected_case_id or getattr(p, "_turn_hints", {}) else "PROCESS_CASE"
            if m.phase == "RESOLVE_INTENT":
                m.selected_case_id = None
            return
        p.add("offer_email", email=mask_email(ph["email"]), repeat=True)
