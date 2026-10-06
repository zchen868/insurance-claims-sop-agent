"""The customer endpoints must expose only the reply, quick replies and a coarse status."""
from fastapi.testclient import TestClient

from sop_agent.server import app

c = TestClient(app)
DEMO = ("I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
        "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.")


def chat(sid, msg):
    r = c.post("/api/client/chat", json={"session_id": sid, "message": msg})
    assert r.status_code == 200
    return r.json()


def test_pages_served():
    assert c.get("/chat").status_code == 200 and "Claims Support" in c.get("/chat").text
    assert c.get("/dev").status_code == 200 and c.get("/").status_code == 200


def test_client_flow_exposes_no_internals():
    v = c.post("/api/client/session").json()
    assert set(v) == {"session_id", "reply", "quick_replies", "status"} and v["status"] == "verifying"
    sid = v["session_id"]
    v = chat(sid, DEMO)
    assert set(v) == {"session_id", "reply", "quick_replies", "status"}
    assert v["status"] == "verified" and "That's all, thanks" in v["quick_replies"]
    v = chat(sid, "That's all, thanks")
    assert v["quick_replies"] == ["Yes, send me the summary", "No thanks, skip it"]
    v = chat(sid, "No thanks, skip it")
    assert v["status"] == "ended" and v["quick_replies"] == []


def test_client_session_not_readable_from_dev_api():
    sid = c.post("/api/client/session").json()["session_id"]
    assert c.get(f"/api/session/{sid}").status_code == 404
