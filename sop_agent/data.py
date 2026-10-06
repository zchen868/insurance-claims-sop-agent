"""Fixture-backed "back office": policyholder lookup, claims, document guidance,
consent requests and an email outbox. Every fact the agent may say about a
claim comes from here, never from the model."""
from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(os.environ.get("SOP_FIXTURES_DIR", ROOT / "fixtures"))

ID_FIELDS = ("full_name", "dob", "phone", "email", "id_last4")
FIELD_LABELS = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number",
    "email": "email address",
    "id_last4": "last 4 digits of your SSN / national ID",
    "policy_number": "policy number",
}


def _load(name):
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


class Backend:
    def __init__(self, source=None):
        from .integrations import claims_source
        self.source = source or claims_source(_load)
        # Business rules / reference data stay local configuration.
        self.guideline = _load("required_document_guideline.json")
        self.schema = _load("claim_schema.json")
        self.consent_scenarios = _load("consent_scenarios.json")
        self.preferred_names = _load("preferred_names.json") if (FIXTURES / "preferred_names.json").exists() else {}

    # Records come from the configured source (fixtures by default, or a claims API).
    @property
    def policyholders(self):
        return self.source.policyholders()

    @property
    def claims(self):
        return self.source.claims()

    @property
    def representatives(self):
        return self.source.representatives()

    def address_name(self, ph):
        """How to address a policyholder: their recorded preferred name, else their full
        name. We never guess which part of a name is the given name."""
        return ph.get("preferred_name") or self.preferred_names.get(ph["party_id"]) or ph["name"]

    # ---------- identity ----------
    @staticmethod
    def norm(field, value):
        if value is None:
            return None
        v = str(value).strip().lower()
        if field == "full_name":
            return re.sub(r"[^a-z ]", "", re.sub(r"\s+", " ", v)).strip()
        if field == "phone":
            digits = re.sub(r"\D", "", v)
            return digits[-10:] if len(digits) >= 10 else digits
        if field == "id_last4":
            return re.sub(r"\D", "", v)[-4:]
        if field == "policy_number":
            return re.sub(r"[^a-z0-9]", "", v)
        return v

    def _values(self, ph, field):
        if field == "full_name":
            return [ph["name"], *ph.get("name_aliases", [])]
        if field == "phone":
            return [ph["phone"], *ph.get("phone_aliases", [])]
        if field == "email":
            return [ph["email"], *ph.get("email_aliases", [])]
        if field == "id_last4":
            return [ph["id_last4"]]
        if field == "policy_number":
            return [ph["policy_number"]]
        return [ph.get(field)]

    def match_identity(self, provided: dict):
        """Score every policyholder against the provided identity fields.
        Returns the best candidate as (party_id, matched_fields, mismatched_fields)."""
        best = None
        for ph in self.policyholders:
            matched, mismatched = [], []
            for field, value in provided.items():
                if not value:
                    continue
                n = self.norm(field, value)
                if n in {self.norm(field, x) for x in self._values(ph, field)}:
                    matched.append(field)
                else:
                    mismatched.append(field)
            score = len([f for f in matched if f in ID_FIELDS]) * 2 + ("policy_number" in matched) - len(mismatched)
            if best is None or score > best[0]:
                best = (score, ph["party_id"], matched, mismatched)
        return best[1:] if best else (None, [], [])

    def policyholder(self, party_id):
        return next((p for p in self.policyholders if p["party_id"] == party_id), None)

    def representative_for(self, rep_name, party_id):
        n = self.norm("full_name", rep_name)
        return next((r for r in self.representatives
                     if r["buyer_party_id"] == party_id and self.norm("full_name", r["rep_name"]) == n), None)

    # ---------- claims ----------
    def claims_for(self, party_id):
        return [c for c in self.claims if c["party_id"] == party_id]

    def claim(self, party_id, case_id):
        return next((c for c in self.claims_for(party_id) if c["case_id"].lower() == str(case_id).lower()), None)

    def find_claims(self, party_id, hints: dict):
        """Rank the caller's claims against remembered hints (type/status/month/year/id)."""
        claims = self.claims_for(party_id)
        if hints.get("case_id"):
            c = self.claim(party_id, hints["case_id"])
            return [c] if c else []
        scored = []
        for c in claims:
            y, m, _ = c["created_at"].split("-")
            checks = []
            if hints.get("case_type"):
                checks.append(c["case_type"] == hints["case_type"])
            if hints.get("status"):
                checks.append(c["status"] == hints["status"])
            if hints.get("month"):
                checks.append(int(m) == int(hints["month"]))
            if hints.get("year"):
                checks.append(int(y) == int(hints["year"]))
            if checks and all(checks):
                scored.append(c)
        return scored

    @staticmethod
    def claim_label(c):
        d = date.fromisoformat(c["created_at"])
        return f"{c['status']} {c['case_type']} claim {c['case_id']} (filed {d.strftime('%B %-d, %Y')})"

    # ---------- grounded knowledge for one claim ----------
    def _doc_key(self, doc):
        keys = self.guideline["document_guidance"].keys()
        d = doc.lower()
        for k in keys:
            if d == k or d in k or k in d:
                return k
        return None

    def case_facts(self, claim, question: str | None = None, today: str | None = None):
        g = self.guideline
        deadline = None
        if claim.get("appeal_deadline"):
            days = (date.fromisoformat(claim["appeal_deadline"]) - date.fromisoformat(today or date.today().isoformat())).days
            deadline = {"appeal_deadline": claim["appeal_deadline"], "today": today, "passed": days < 0,
                        "days_left": max(days, 0)}
        docs = claim.get("documents_needed", [])
        doc_text = ", ".join(docs) if docs else "the requested documents"
        fill = {
            "case_id": claim["case_id"],
            "documents": doc_text,
            "average_processing_time_after_submission":
                g["claim_followup_settings"]["average_processing_time_after_submission"]["en"],
        }
        documents = []
        for d in docs:
            k = self._doc_key(d)
            documents.append({
                "document": d,
                "how_to_prepare": g["document_guidance"].get(k, {}).get("en"),
                "if_unavailable": g["document_alternative_guidance"].get(k, g["document_alternative_guidance"]["default"])["en"],
            })
        followups, matched_topics = [], []
        q = (question or "").lower()
        for item in g["claim_followup_guidance"]:
            if item.get("requires_documents") and not docs:
                continue
            if deadline and deadline["passed"] and item["topic"] == "submission_timing":
                continue  # "submit within a week" no longer applies once the appeal window closed
            text = item["en"].format(**fill)
            followups.append({"topic": item["topic"], "text": text})
            if item.get("match_any") and any(p in q for p in item["match_any"]):
                matched_topics.append(item["topic"])
        facts = {
            "claim": claim,
            "field_meanings": {k: v["description"] for k, v in self.schema["field_descriptions"].items()},
            "general_submission_guidance": g["default_guidance"]["en"],
            "case_type_guidance": g["case_type_guidance"].get(claim["case_type"], {}).get("en"),
            "documents": documents,
            "followup_guidance": followups,
            "matched_followup_topics": matched_topics,
            "followup_fallback": g["claim_followup_fallback"]["en"],
            "human_review_policy": g["claim_followup_settings"]["human_review_after_document_alternatives_exhausted"]["en"],
            "appeal_deadline_status": deadline,
        }
        if deadline and deadline["passed"]:
            facts["late_appeal_policy"] = ("The appeal deadline for this claim has passed. The agent cannot accept or "
                                           "promise a late appeal; a human claims representative must review whether "
                                           "late documents can still be considered.")
        return facts

    # ---------- side-effecting tools ----------
    def consent_sequence(self, scenario):
        return self.consent_scenarios.get(scenario, self.consent_scenarios["default"])["status_sequence"]

    def send_email(self, to, subject, body):
        from .integrations import email_sender
        return email_sender().send(to, subject, body)


def mask_email(e):
    user, _, dom = e.partition("@")
    return f"{user[0]}{'*' * max(len(user) - 1, 3)}@{dom}"


BACKEND = Backend()
