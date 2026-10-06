"""One conversation = one Session. A turn runs:
   understand (NLU) -> remember + decide (SOP engine) -> act (tools) -> phrase (LLM) -> guard."""
from __future__ import annotations

import os
import re
import time
import uuid
from datetime import date

from . import nlu, responder
from .data import BACKEND
from .llm import build_llm
from .memory import Memory
from .sop import Engine
from .store import get_store

GREETING = ("Hi, thanks for contacting Claims Support. I'm Ava, a virtual assistant. I can help with questions about "
            "your insurance claims. To get started, could you tell me your full name and a couple of other details, "
            "such as your date of birth, phone number, email, or the last 4 digits of your SSN?")


def integrations_summary():
    """Which real vs demo connections are active (shown in the developer view)."""
    from .integrations import claims_source, email_sender, sms_sender
    return {"claims_data": BACKEND.source.name, "email": email_sender().name, "sms": sms_sender().name,
            "consent": os.environ.get("CONSENT_MODE", "scenario")}


SSN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?(\d{4})\b")


def redact_ssn(text):
    """A full SSN is never stored, logged or sent to the model: keep only the last 4."""
    out = SSN.sub(lambda mm: f"***-**-{mm.group(1)}", text)
    return out, out != text


class Session:
    def __init__(self, api_key: str | None = None, consent_scenario: str = "default", *,
                 kind: str = "dev", store=None, today: str | None = None, sid: str | None = None):
        self.id = sid or uuid.uuid4().hex[:12]
        self.kind = kind
        self.store = store if store is not None else get_store()
        self.memory = Memory(consent_scenario=consent_scenario)
        self.custom_key = bool(api_key)     # a UI-supplied key is never persisted
        self.llm = build_llm(api_key)
        self.today = today or os.environ.get("SOP_TODAY", date.today().isoformat())
        self.engine = Engine(today=self.today, store=self.store, session_id=self.id)
        self.history = [{"role": "agent", "text": GREETING}]
        self.turns = []
        self.save()

    # ---------- persistence ----------
    def save(self):
        if self.store:
            self.store.save_session(self.id, self.kind, {
                "memory": self.memory.to_dict(), "history": self.history, "turns": self.turns[-50:],
                "today": self.today, "custom_key": self.custom_key})

    @classmethod
    def load(cls, sid: str, kind: str, store=None):
        store = store if store is not None else get_store()
        d = store.load_session(sid, kind)
        if d is None:
            return None
        s = cls(kind=kind, store=store, today=d["today"], sid=sid)
        mem = {k: v for k, v in d["memory"].items() if k in Memory.__dataclass_fields__}
        s.memory = Memory(**mem)
        s.history, s.turns = d["history"], d["turns"]
        return s

    @property
    def mode(self):
        return self.llm.label if self.llm else "offline (rule-based)"

    def turn(self, text: str) -> dict:
        t0 = time.time()
        m = self.memory
        last_agent = self.history[-1]["text"] if self.history and self.history[-1]["role"] == "agent" else ""
        text, full_ssn = redact_ssn(text)
        x, nlu_source = nlu.extract(self.llm, text, m.phase, last_agent, self.today)
        x["full_ssn_shared"] = full_ssn
        self.history.append({"role": "caller", "text": text})

        plan = self.engine.step(m, x, text)

        # Side-effecting tool calls happen only when the engine plans them.
        for a in plan.acts:
            if a["type"] == "send_email":
                body, src = responder.compose_summary(self.llm, m)
                rec = BACKEND.send_email(a["to"], "Summary of your Claims Support conversation", body)
                rec["composed_by"] = src
                m.email_record = rec
                plan.trace.append(f"tool: send_email -> {a['masked']} (id {rec['id']})")

        reply, reply_source, violations = None, "template", []
        if self.llm is not None:
            draft = responder.llm_render(self.llm, plan, m, self.history[:-1], text)
            if draft:
                violations = responder.guard(draft, plan, m)
                if not violations:
                    reply, reply_source = draft, "llm"
                else:
                    plan.trace.append("guard blocked LLM draft: " + "; ".join(violations))
            else:
                plan.trace.append(f"LLM reply unavailable ({self.llm.last_error}); using template")
        if reply is None:
            reply = responder.render_template(plan, m)

        self.history.append({"role": "agent", "text": reply})
        rec = {
            "caller": text, "agent": reply, "phase_before": plan.phase_before, "phase_after": m.phase,
            "extraction": x, "nlu_source": nlu_source, "acts": [a["type"] for a in plan.acts],
            "reply_source": reply_source, "guard_violations": violations, "trace": plan.trace,
            "latency_ms": int((time.time() - t0) * 1000),
        }
        self.turns.append(rec)
        self.save()
        return rec

    def state(self):
        return {"session_id": self.id, "mode": self.mode, "memory": self.memory.to_dict(),
                "integrations": integrations_summary(),
                "history": self.history, "turns": self.turns}
