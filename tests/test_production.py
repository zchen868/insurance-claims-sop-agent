"""Production-readiness features: persistence, cross-chat lockout, real integrations
(link consent, SMTP, Twilio, claims API), name handling, dev auth, rate limiting."""
import glob
import json
import os

import httpx
from fastapi.testclient import TestClient

from sop_agent import server
from sop_agent.agent import Session
from sop_agent.data import Backend, _load
from sop_agent.integrations import HttpClaimsSource
from sop_agent.sop import Engine
from sop_agent.store import Store

DEMO = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
        "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")
REP = ("My name is David Chen, I'm calling on behalf of my mother Margaret Chen. "
       "Her DOB is 1985-03-15, phone 650-521-2836, SSN last four 4472.")
c = TestClient(server.app)


# ---------------------------------------------------------------- persistence
def test_session_survives_restart():
    s = Session(kind="client")
    s.turn("Hi, I'm calling about my denied healthcare claim from January.")
    restored = Session.load(s.id, "client")
    assert restored.memory.case_hints == s.memory.case_hints and len(restored.history) == len(s.history)
    restored.turn("Margaret Chen, DOB 1985-03-15, SSN last 4 is 4472")
    assert restored.memory.selected_case_id == "CL-2048"     # remembered hint still used after "restart"


def test_client_chat_continues_after_server_cache_cleared():
    sid = c.post("/api/client/session").json()["session_id"]
    c.post("/api/client/chat", json={"session_id": sid, "message": "Margaret Chen, DOB 1985-03-15"})
    server.CLIENT_SESSIONS.clear()                            # simulate a process restart
    v = c.post("/api/client/chat", json={"session_id": sid, "message": "SSN last four 4472"}).json()
    assert v["status"] == "verified"


def test_dev_and_client_sessions_stay_separate_in_store():
    s = Session(kind="client")
    assert Session.load(s.id, "dev") is None


# ---------------------------------------------------------------- lockout across chats
def test_lockout_across_chats():
    bad = "Margaret Chen, DOB 1985-03-15, SSN last four 1111"
    for _ in range(2):                       # 3 + 2 failed attempts spread over chats
        s = Session()
        s.turn(bad)
        s.turn(bad)
    s = Session()
    s.turn(bad)
    s = Session()
    s.turn("Margaret Chen, DOB 1985-03-15, SSN last four 4472")   # correct, but the policy is locked
    assert s.memory.phase == "HANDOFF" and s.memory.verified_party_id is None
    assert "locked" in s.memory.handoff_reason


def test_successful_verification_resets_counter():
    s = Session()
    s.turn("Margaret Chen, DOB 1985-03-15, SSN last four 1111")
    s.turn("sorry, SSN last four 4472")
    assert s.memory.verified_party_id == "P9"
    assert s.store.recent_failures("P9", 3600) == 0


# ---------------------------------------------------------------- link consent
def _sms_for(token):
    for f in glob.glob(os.path.join(os.environ["SOP_OUTBOX_DIR"], "sms-*.json")):
        rec = json.load(open(f))
        if token in rec["body"]:
            return rec
    return None


def test_link_consent_approve():
    s = Session(consent_scenario="link")
    s.turn(REP)
    m = s.memory
    assert "consent_requested" in s.turns[-1]["acts"] and m.consent_ref
    sms = _sms_for(m.consent_ref)
    assert sms and sms["to"] == "+16505212836" and "David Chen" in sms["body"]
    s.turn("has she approved yet?")
    assert "consent_pending" in s.turns[-1]["acts"]
    page = c.get(f"/consent/{m.consent_ref}")
    assert page.status_code == 200 and "Approve" in page.text and "POL-9921" in page.text
    assert c.post(f"/api/consent/{m.consent_ref}", json={"decision": "approve"}).status_code == 200
    s.turn("ok she approved it")
    assert m.gate_open and "consent_approved" in s.turns[-1]["acts"]
    assert c.post(f"/api/consent/{m.consent_ref}", json={"decision": "decline"}).status_code == 409


def test_link_consent_decline():
    s = Session(consent_scenario="link")
    s.turn(REP)
    c.post(f"/api/consent/{s.memory.consent_ref}", json={"decision": "decline"})
    s.turn("is it approved?")
    assert "consent_declined" in s.turns[-1]["acts"] and not s.memory.gate_open


def test_link_consent_timeout(monkeypatch):
    monkeypatch.setenv("CONSENT_TIMEOUT_MIN", "0")
    s = Session(consent_scenario="link")
    s.turn(REP)
    s.turn("any update?")
    assert s.memory.consent_status == "timeout" and "offer_human" in s.turns[-1]["acts"]


def test_bad_consent_link():
    assert "invalid" in c.get("/consent/not-a-token").text
    assert c.post("/api/consent/not-a-token", json={"decision": "approve"}).status_code == 409


# ---------------------------------------------------------------- email / sms providers
def test_smtp_email(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            sent.append(("starttls",))

        def login(self, u, p):
            sent.append(("login", u))

        def send_message(self, msg):
            sent.append(("send", msg["To"], msg["Subject"]))

    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    s = Session()
    for t in (DEMO, "that's all", "yes please send it"):
        s.turn(t)
    assert ("send", "margaret@email.com", "Summary of your Claims Support conversation") in sent
    assert ("starttls",) in sent and s.memory.email_record["delivered_via"].startswith("smtp")


def test_twilio_sms(monkeypatch):
    calls = []
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    monkeypatch.setenv("TWILIO_FROM", "+15550000000")
    monkeypatch.setattr(httpx, "post", lambda url, data, auth, timeout: calls.append((url, data)) or httpx.Response(201))
    s = Session(consent_scenario="link")
    s.turn(REP)
    assert calls and calls[0][0].endswith("/Accounts/AC123/Messages.json") and calls[0][1]["To"] == "+16505212836"


# ---------------------------------------------------------------- claims API
def _mock_claims_api(counter):
    data = {"/policyholders": _load("policyholders.json"), "/claims": _load("claims.json"),
            "/representatives": _load("representatives.json")}

    def handler(req):
        counter.append(req.url.path)
        assert req.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json=data[req.url.path])
    return httpx.MockTransport(handler)


def test_http_claims_source_drives_the_engine():
    hits = []
    src = HttpClaimsSource(base_url="https://claims.example.com", token="secret", ttl=60, transport=_mock_claims_api(hits))
    b = Backend(source=src)
    pid, matched, mismatched = b.match_identity({"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"})
    assert pid == "P9" and not mismatched
    assert [c_["case_id"] for c_ in b.find_claims("P9", {"status": "denied"})] == ["CL-2048"]
    b.claims_for("P9")
    assert hits.count("/claims") == 1                          # cached within TTL
    e = Engine(backend=b, today="2026-03-05")
    from sop_agent.memory import Memory
    from sop_agent.nlu import rule_extract
    m = Memory()
    plan = e.step(m, rule_extract(DEMO, "VERIFY_ID", ""), DEMO)
    assert m.selected_case_id == "CL-2048" and "answer" in [a["type"] for a in plan.acts]


# ---------------------------------------------------------------- names
def test_preferred_name_from_records():
    s = Session()
    s.turn(DEMO)
    assert s.memory.address_as == "Margaret"


def test_full_name_when_no_preference_known():
    s = Session()
    s.turn("Ma Tian, DOB 1964-09-10, national ID last four 6688")
    assert s.memory.address_as == "Ma Tian" and "Ma Tian" in s.turns[-1]["agent"]


def test_caller_stated_preferred_name():
    s = Session()
    s.turn("Ma Tian, DOB 1964-09-10, national ID last four 6688. Please call me Tian.")
    assert s.memory.address_as == "Tian"


# ---------------------------------------------------------------- hosting protections
def test_dev_view_password(monkeypatch):
    monkeypatch.setenv("DEV_PASSWORD", "s3cret")
    assert c.get("/dev").status_code == 401
    assert c.post("/api/session", json={}).status_code == 401
    assert c.get("/dev", auth=("dev", "s3cret")).status_code == 200
    assert c.get("/chat").status_code == 200                   # customers never need it


def test_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    server._hits.clear()
    codes = [c.post("/api/client/session", headers={"x-forwarded-for": "9.9.9.9"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    server._hits.clear()
