"""FastAPI app.

  /      and /dev   developer view: chat + live SOP internals (phase, memory, trace)
  /chat             customer view: clean chat only
  /consent/{token}  page a policyholder opens from the SMS to approve or decline a representative

Production switches (all optional): DEV_PASSWORD protects the developer view and API
with HTTP Basic auth; RATE_LIMIT_PER_MIN throttles the customer chat per client IP;
conversations persist in SQLite (SOP_DB_PATH) so restarts don't drop live chats.

Customer sessions live in their own store and their endpoints return only what a
customer may see (the reply, quick-reply buttons, a coarse status), never memory,
identity fields or the engine trace."""
from __future__ import annotations

import html
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from .agent import Session, integrations_summary
from .data import BACKEND
from .store import get_store

@asynccontextmanager
async def lifespan(_app):
    get_store().purge_sessions(older_than_s=float(os.environ.get("SESSION_TTL_HOURS", "24")) * 3600)
    yield


app = FastAPI(title="Insurance Claims SOP Agent", lifespan=lifespan)
SESSIONS: dict[str, Session] = {}
STATIC = Path(__file__).parent / "static"
_basic = HTTPBasic(auto_error=False)


def dev_auth(creds: HTTPBasicCredentials | None = Depends(_basic)):
    """Developer view exposes internals: require a password when DEV_PASSWORD is set (always set it when hosted)."""
    pw = os.environ.get("DEV_PASSWORD")
    if not pw:
        return
    if not creds or not (secrets.compare_digest(creds.username, os.environ.get("DEV_USER", "dev"))
                         and secrets.compare_digest(creds.password, pw)):
        raise HTTPException(401, "Developer view requires a password", headers={"WWW-Authenticate": "Basic"})


_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request):
    limit = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    q, now = _hits[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "You're sending messages too quickly. Please wait a moment.")
    q.append(now)


def _session(cache: dict, sid: str, kind: str) -> Session | None:
    s = cache.get(sid)
    if s is None:
        s = Session.load(sid, kind)      # survives restarts
        if s is not None:
            cache[sid] = s
    return s



class NewSession(BaseModel):
    api_key: str | None = None
    consent_scenario: str = "default"


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.get("/", dependencies=[Depends(dev_auth)])
@app.get("/dev", dependencies=[Depends(dev_auth)])
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/chat")
def client_page():
    return FileResponse(STATIC / "client.html")


@app.get("/api/health")
def health():
    return {"ok": True, "provider": os.environ.get("AI_PROVIDER", "anthropic"), "integrations": integrations_summary(),
            "server_key_configured": bool(os.environ.get("AI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
                                          or os.environ.get("ANTHROPIC_AUTH_TOKEN"))}


@app.post("/api/session", dependencies=[Depends(dev_auth)])
def new_session(body: NewSession):
    s = Session(api_key=(body.api_key or "").strip() or None, consent_scenario=body.consent_scenario, kind="dev")
    SESSIONS[s.id] = s
    return s.state()


@app.post("/api/chat", dependencies=[Depends(dev_auth)])
def chat(body: ChatIn):
    s = _session(SESSIONS, body.session_id, "dev")
    if not s:
        raise HTTPException(404, "Unknown session; start a new one.")
    msg = body.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    turn = s.turn(msg[:2000])
    return {"turn": turn, "state": s.state()}


@app.get("/api/session/{sid}", dependencies=[Depends(dev_auth)])
def get_session(sid: str):
    s = _session(SESSIONS, sid, "dev")
    if not s:
        raise HTTPException(404, "Unknown session")
    return s.state()


# ---------------------------------------------------------------- customer view
CLIENT_SESSIONS: dict[str, Session] = {}


def _client_view(s: Session, reply: str | None = None) -> dict:
    m = s.memory
    if m.phase == "HANDOFF":
        status, quick = "handoff", []
    elif m.phase == "ENDED":
        status, quick = "ended", []
    else:
        status = "verified" if m.gate_open else ("consent" if m.verified_party_id else "verifying")
        q = m.last_agent_question
        if m.phase == "POST_PROCESS" and m.email_state == "offered":
            quick = ["Yes, send me the summary", "No thanks, skip it"]
        elif q == "offer_human" and m.human_offered:
            quick = ["Yes, transfer me", "No, let's continue"]
        elif m.phase == "RESOLVE_INTENT" and m.gate_open and q in ("pick_case", "intent"):
            ids = m.candidate_case_ids or [c["case_id"] for c in BACKEND.claims_for(m.verified_party_id)]
            quick = [f"Claim {cid}" for cid in ids]
        elif m.phase == "PROCESS_CASE" and q == "anything_else":
            quick = ["What are my next steps?", "That's all, thanks"]
        else:
            quick = []
    return {"session_id": s.id, "reply": reply if reply is not None else s.history[-1]["text"],
            "quick_replies": quick, "status": status}


@app.post("/api/client/session", dependencies=[Depends(rate_limit)])
def client_new_session():
    s = Session(kind="client")
    CLIENT_SESSIONS[s.id] = s
    return _client_view(s)


@app.post("/api/client/chat", dependencies=[Depends(rate_limit)])
def client_chat(body: ChatIn):
    s = _session(CLIENT_SESSIONS, body.session_id, "client")
    if not s:
        raise HTTPException(404, "Your session has expired. Please start a new chat.")
    msg = body.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    turn = s.turn(msg[:2000])
    return _client_view(s, turn["agent"])


# ---------------------------------------------------------------- consent link
class ConsentIn(BaseModel):
    decision: str   # approve | decline


def _consent_page(title, body, buttons=""):
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Claims Support consent</title>
<style>
:root {{ --bg:#eef1f6; --card:#fff; --text:#1b2230; --muted:#6b7484; --brand:#1f4fd1; --line:#e4e8ee; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1218; --card:#181c24; --text:#e7e9ee; --muted:#9aa3b2; --brand:#6f8fff; --line:#2a303b; }} }}
body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui,-apple-system,sans-serif;
  display:flex; min-height:100vh; align-items:center; justify-content:center; padding:16px; box-sizing:border-box; }}
.card {{ background:var(--card); border-radius:16px; padding:28px; max-width:440px; width:100%; box-shadow:0 10px 40px rgba(0,0,0,.1); }}
h1 {{ font-size:20px; margin:0 0 12px; }} p {{ color:var(--muted); margin:0 0 20px; }}
.row {{ display:flex; gap:10px; }}
button {{ flex:1; font:inherit; font-weight:600; padding:12px; border-radius:10px; cursor:pointer; border:1px solid var(--brand); }}
.yes {{ background:var(--brand); color:#fff; }} .no {{ background:transparent; color:var(--brand); }}
</style></head><body><div class="card"><h1>{title}</h1><p>{body}</p>{buttons}</div>
<script>
async function decide(d) {{
  const r = await fetch(location.pathname.replace("/consent/", "/api/consent/"), {{ method: "POST",
    headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify({{ decision: d }}) }});
  const v = await r.json();
  document.querySelector(".card").innerHTML = "<h1>" + (r.ok ? (d === "approve" ? "Access approved" : "Request declined") : "Link no longer valid")
    + "</h1><p>" + (r.ok ? "You can close this page." : (v.detail || "")) + "</p>";
}}
</script></body></html>""")


@app.get("/consent/{token}")
def consent_page(token: str):
    c = get_store().get_consent(token)
    if not c:
        return _consent_page("Link not found", "This consent link is invalid.")
    if c["status"] != "pending":
        return _consent_page("Already answered", f"This request was already {html.escape(c['status'])}.")
    ph = BACKEND.policyholder(c["party_id"])
    who = html.escape(c["rep_name"] or "Someone")
    return _consent_page(
        f"Allow {who} to discuss your claims?",
        f"{who} contacted Claims Support on your behalf and asked to discuss the claims on policy "
        f"{html.escape(ph['policy_number'])}. Only approve if you know and trust this person. "
        "Approval applies to this conversation only.",
        '<div class="row"><button class="yes" onclick="decide(\'approve\')">Approve</button>'
        '<button class="no" onclick="decide(\'decline\')">Decline</button></div>')


@app.post("/api/consent/{token}")
def consent_decide(token: str, body: ConsentIn):
    if body.decision not in ("approve", "decline"):
        raise HTTPException(400, "decision must be approve or decline")
    ok = get_store().decide_consent(token, "approved" if body.decision == "approve" else "declined")
    if not ok:
        raise HTTPException(409, "This request was already answered or has expired.")
    return {"ok": True}
