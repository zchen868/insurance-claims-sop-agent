"""Pluggable connections to the outside world. Each has a local/demo
implementation (the default) and a real one switched on by environment variables:

  Email    OutboxEmail (writes outbox/*.json)        | SmtpEmail     (SMTP_HOST, ...)
  SMS      OutboxSms   (writes outbox/sms-*.json)    | TwilioSms     (TWILIO_ACCOUNT_SID, ...)
  Consent  ScenarioConsent (fixture status sequence) | LinkConsent   (CONSENT_MODE=link: SMS link the
                                                                      policyholder approves or declines)
  Claims   FixtureSource (fixtures/*.json)           | HttpClaimsSource (CLAIMS_API_URL, CLAIMS_API_TOKEN)

The SOP engine only talks to these interfaces, so swapping mock for real changes
no workflow logic."""
from __future__ import annotations

import json
import logging
import os
import secrets
import smtplib
import time
import uuid
from email.message import EmailMessage
from pathlib import Path

import httpx

log = logging.getLogger("sop.integrations")
ROOT = Path(__file__).resolve().parent.parent


def outbox_dir() -> Path:
    d = Path(os.environ.get("SOP_OUTBOX_DIR", ROOT / "outbox"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_outbox(prefix, rec):
    with open(outbox_dir() / f"{prefix}{rec['id']}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)


# ================================================================ email
class OutboxEmail:
    name = "outbox"

    def send(self, to, subject, body):
        rec = {"id": uuid.uuid4().hex[:10], "to": to, "subject": subject, "body": body,
               "sent_at": time.strftime("%Y-%m-%d %H:%M:%S"), "delivered_via": "outbox (demo)"}
        _write_outbox("", rec)
        return rec


class SmtpEmail:
    name = "smtp"

    def __init__(self):
        self.host = os.environ["SMTP_HOST"]
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.user = os.environ.get("SMTP_USER")
        self.password = os.environ.get("SMTP_PASSWORD")
        self.sender = os.environ.get("SMTP_FROM", self.user or "claims-support@localhost")
        self.starttls = os.environ.get("SMTP_STARTTLS", "true").lower() != "false"

    def send(self, to, subject, body):
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        rec = {"id": uuid.uuid4().hex[:10], "to": to, "subject": subject, "body": body,
               "sent_at": time.strftime("%Y-%m-%d %H:%M:%S"), "delivered_via": f"smtp ({self.host})"}
        try:
            with smtplib.SMTP(self.host, self.port, timeout=20) as s:
                if self.starttls:
                    s.starttls()
                if self.user:
                    s.login(self.user, self.password or "")
                s.send_message(msg)
        except (smtplib.SMTPException, OSError) as e:
            rec["delivered_via"] = f"FAILED: {e}"
            log.error("SMTP send failed: %s", e)
        _write_outbox("", rec)   # audit copy
        return rec


def email_sender():
    return SmtpEmail() if os.environ.get("SMTP_HOST") else OutboxEmail()


# ================================================================ SMS
class OutboxSms:
    name = "outbox"

    def send(self, to, body):
        rec = {"id": uuid.uuid4().hex[:10], "to": to, "body": body, "sent_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "delivered_via": "outbox (demo)"}
        _write_outbox("sms-", rec)
        return rec


class TwilioSms:
    name = "twilio"

    def __init__(self):
        self.sid = os.environ["TWILIO_ACCOUNT_SID"]
        self.token = os.environ["TWILIO_AUTH_TOKEN"]
        self.sender = os.environ["TWILIO_FROM"]

    def send(self, to, body):
        rec = {"id": uuid.uuid4().hex[:10], "to": to, "body": body, "sent_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            r = httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json",
                           data={"To": to, "From": self.sender, "Body": body}, auth=(self.sid, self.token), timeout=20)
            rec["delivered_via"] = f"twilio ({r.status_code})"
        except httpx.HTTPError as e:
            rec["delivered_via"] = f"FAILED: {e}"
        _write_outbox("sms-", rec)
        return rec


def sms_sender():
    return TwilioSms() if os.environ.get("TWILIO_ACCOUNT_SID") else OutboxSms()


# ================================================================ consent
class ScenarioConsent:
    """Demo/test consent: status follows a fixture sequence, one step per caller turn."""
    name = "scenario"

    def __init__(self, backend, scenario="default"):
        self.seq = backend.consent_sequence(scenario)

    def request(self, m, ph, session_id):
        m.consent_polls = 1
        return {"status": self.seq[0], "ref": None, "link": None}

    def status(self, m):
        status = self.seq[min(m.consent_polls, len(self.seq) - 1)]
        m.consent_polls += 1
        if status == "pending" and m.consent_polls >= len(self.seq):
            return "timeout"
        return status


class LinkConsent:
    """Real consent: the policyholder gets an SMS with a one-time link and approves or
    declines on their own phone. The chat polls the stored decision each turn."""
    name = "link"

    def __init__(self, store, sms=None):
        self.store = store
        self.sms = sms or sms_sender()
        self.timeout_s = float(os.environ.get("CONSENT_TIMEOUT_MIN", "10")) * 60

    def request(self, m, ph, session_id):
        token = secrets.token_urlsafe(16)
        self.store.create_consent(token, session_id, ph["party_id"], m.rep_name)
        base = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
        link = f"{base}/consent/{token}"
        self.sms.send(ph["phone"], f"Claims Support: {m.rep_name} is asking to discuss your insurance claims on your "
                                   f"behalf. Approve or decline here: {link}")
        m.consent_polls = 1
        return {"status": "pending", "ref": token, "link": link}

    def status(self, m):
        c = self.store.get_consent(m.consent_ref) if m.consent_ref else None
        m.consent_polls += 1
        if not c:
            return "timeout"
        if c["status"] == "pending" and time.time() - c["created"] > self.timeout_s:
            return "timeout"
        return c["status"]


def consent_service(backend, store, scenario):
    if os.environ.get("CONSENT_MODE", "scenario") == "link" or scenario == "link":
        return LinkConsent(store)
    return ScenarioConsent(backend, scenario)


# ================================================================ claims data
class FixtureSource:
    """The sample data shipped with the assignment."""
    name = "fixtures"

    def __init__(self, load):
        self._load = load

    def policyholders(self):
        return self._load("policyholders.json")

    def claims(self):
        return self._load("claims.json")

    def representatives(self):
        return self._load("representatives.json")


class HttpClaimsSource:
    """A claims/policy system exposed over REST. Expected contract (JSON):
         GET {CLAIMS_API_URL}/policyholders    -> [policyholder, ...]  (same fields as fixtures)
         GET {CLAIMS_API_URL}/claims           -> [claim, ...]
         GET {CLAIMS_API_URL}/representatives  -> [representative, ...]
       Authorization: Bearer {CLAIMS_API_TOKEN}. Responses are cached for CLAIMS_API_TTL seconds."""
    name = "http"

    def __init__(self, base_url=None, token=None, ttl=None, transport=None):
        self.base = (base_url or os.environ["CLAIMS_API_URL"]).rstrip("/")
        token = token or os.environ.get("CLAIMS_API_TOKEN")
        self.ttl = float(ttl if ttl is not None else os.environ.get("CLAIMS_API_TTL", "60"))
        self.http = httpx.Client(timeout=15, transport=transport,
                                 headers={"Authorization": f"Bearer {token}"} if token else {})
        self._cache: dict[str, tuple[float, list]] = {}

    def _get(self, path):
        hit = self._cache.get(path)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        r = self.http.get(self.base + path)
        r.raise_for_status()
        data = r.json()
        self._cache[path] = (time.time(), data)
        return data

    def policyholders(self):
        return self._get("/policyholders")

    def claims(self):
        return self._get("/claims")

    def representatives(self):
        return self._get("/representatives")


def claims_source(load):
    return HttpClaimsSource() if os.environ.get("CLAIMS_API_URL") else FixtureSource(load)
