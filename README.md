# Insurance Claims SOP Agent

Repository: https://github.com/zchen868/insurance-claims-sop-agent

This submission provides a Docker-ready repository with setup instructions below. Configure your own AI model API token to run the natural-language demo.

## Assessment requirement coverage

- **Strict identity gate:** at least three matching fields from full name, DOB, phone, email, and SSN last four. Policy number is a lookup hint and does not count toward the three. Partial answers accumulate; failed verification never opens the disclosure gate.
- **Memory across phases:** claim intent and hints are captured during VERIFY_ID and reused after verification. The supplied Margaret Chen sample resolves to CL-2048 without restarting intent discovery.
- **Bounded model reasoning:** the real LLM interprets natural language, ambiguity, follow-up questions and emotion, and phrases grounded answers. Deterministic handlers choose permitted transitions and tool actions from those interpretations. The model cannot directly execute arbitrary tools or bypass verification. Output checks cover protected details, claim IDs, amounts and dates; they are not a formal guarantee against every possible hallucination.
- **Scope and recovery:** unrelated questions such as “What is RL?” are declined. The third consecutive irrelevant turn offers a human; the fourth enters simulated HANDOFF.
- **Post-case choice:** offer a summary containing discussed topics, claim status and next steps. Send only after consent, or skip without producing an email. Sending is simulated in the outbox by default.
- **Emotional support bonus:** recognize negative emotions and refusals; place empathy before workflow prompts; explain verification/consent; offer alternate identity fields; offer or simulate human handoff when persuasion should stop.
- **Agreed demo scope:** real LLM calls are required and were demonstrated. No real email delivery, document upload, external support ticket, or connection to a live human is required. Handoff is a demo state and message, with an internal note.

Additional reviewer checks: start a new conversation for each example. Try a claim hint before supplying identity, “What is RL?” repeatedly, the frustrated-caller sample, “I want to speak to a human representative,” and choosing “skip” at the email offer. Inspect the gate, planned acts, and trace in `/dev`.

## Reviewer setup and full-workflow demo

Requires Docker Desktop (running) and an API token for a supported AI model provider.

1. Clone or download this repository and open a terminal in its root directory.
2. Run `cp .env.example .env`.
3. Edit `.env`: set `AI_API_KEY`, `AI_BASE_URL`, `AI_MODEL`, and `AI_API_FORMAT` for your provider. The bundled example uses DeepSeek with the Responses API. Use the model and endpoint supported by your account. Never commit `.env`.
4. Run `docker compose up --build -d`.
5. Open http://localhost:8000/dev (reviewer view) or http://localhost:8000/chat (customer view).

If port 8000 is occupied, change the Compose port mapping from `8000:8000` to `8001:8000` and `PUBLIC_BASE_URL` to `http://localhost:8001`, then use port 8001 in your browser.

### Demonstrate all four stages

In `/dev`, click **New conversation**, then **Demo test case**. No key needs to be entered in the UI when `.env` is configured. The test message supplies fixture identity details and asks about a denied January healthcare claim. The engine verifies Margaret Chen, resolves her intent, selects CL-2048, and answers the claim question. These transitions are visible in the engine trace.

Send these messages one at a time, waiting for each response:

```text
What are my next steps?
That's all, thanks.
Yes, send me the summary.
```

Expected result: the agent offers a summary during POST_PROCESS, writes the simulated email to the outbox, and finishes in ENDED with Email = sent. `Understood by: llm` and `Reply by: llm` show that the AI model handled those turns. Without a key, or when model calls fail, rules/templates may be used; check the developer trace rather than treating a successful health check as evidence of model success.

This is a demo using fixture records. Email/SMS are simulated unless real integrations are configured. `SOP_TODAY=2026-03-05` is a fixed demonstration date matching the fixture scenarios. SQLite and the outbox persist in the Docker named volume. Keep one service instance.

Stop the service with `docker compose down` (the named volume is retained).

### Validation

A local Docker build and model-backed UI walkthrough were completed on October 6, 2026: VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS → ENDED, with an LLM-composed summary in the simulated outbox. The submitted code also passed 57/57 automated tests and 49/49 offline evaluation scenarios (103 turns) in an isolated Docker container on October 6, 2026. Offline checks validate workflow behavior; they do not measure live-model quality.


A claims-support chat agent that follows a fixed business workflow but still talks naturally.

```
VERIFY_ID ──► RESOLVE_INTENT ──► PROCESS_CASE ──► POST_PROCESS ──► ENDED
    └──────────────── HANDOFF (human representative) ◄──────────────┘
```

## Quick start

### Configure the model (API token)

Copy `.env.example` to `.env` and fill in one provider:

```bash
# OpenAI-compatible provider, e.g. DeepSeek (tested with deepseek-flash, Responses API)
AI_PROVIDER=deepseek
AI_BASE_URL=https://api.deepseek.com
AI_API_KEY=sk-...
AI_MODEL=deepseek-flash
AI_API_FORMAT=responses     # or "chat" for /chat/completions

# or Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...   # SOP_MODEL defaults to claude-opus-5-5
```

You can also paste a key into the field in the UI header and click **New conversation**. It overrides the configured key for that session only and is never written to disk.

If no token is configured, the app runs in **offline mode**, using rule-based understanding and template replies. The SOP behaves the same way; only the wording is less flexible. The badge in the header shows the active backend.

### Run with Docker

```bash
docker build -t claims-sop-agent .
docker run -p 8000:8000 --env-file .env claims-sop-agent
# open http://localhost:8000
```

### Run locally (Python 3.10+)

```bash
./run.sh                    # creates .venv, installs deps, loads .env, serves http://localhost:8000
```

### Tests

```bash
.venv/bin/python -m pytest -q              # 57 unit tests, offline, ~1 s
.venv/bin/python -m evals.run_eval --offline   # 49-scenario evaluation suite, no API calls
.venv/bin/python -m evals.run_eval --judge --repeat 2   # live model + LLM judge, each scenario twice
```

The unit tests cover the SOP rules (verification, memory, intent, processing, post-process, representatives, safety, the output guard) and the production features (persistence, cross-chat lockout, consent links, SMTP, Twilio, claims API, names, developer-view password, rate limiting).

## Two views

| URL | Who it's for | What it shows |
|---|---|---|
| `http://localhost:8000/chat` | **Customers** | A clean chat: a verification status banner, quick-reply buttons when the bot asks for a choice (send/skip the email, which claim, transfer or not), a "Talk to a person" button, and a clear ended or transferred state. It uses a separate, restricted API (`/api/client/*`) that returns only the reply, the buttons and a coarse status, never memory, identity fields or the engine trace. |
| `http://localhost:8000/` or `/dev` | **Developers / reviewers** | The chat plus the SOP internals described below. |

## Developer view

- **Chat pane**: type as the caller. The chips under the box hold ready-made test messages (the demo test case, a frustrated caller, an out-of-scope question, a representative, …).
- **Side panel**: shows what's happening inside the SOP engine:
  - the current phase and whether claim details are unlocked (the *gate*)
  - working memory: identity fields heard, remembered intent and case hints, selected case, emotion trend, email state
  - the last turn's plan (*acts*), engine trace, whether the LLM or the rules understood the message, whether the LLM or a template wrote the reply, and any guard intervention
  - the email outbox (the sent summary) and the human-handoff note
- **Consent dropdown**: picks the fixture scenario for representative callers (`default` approves, `timeout` never does).

Sent emails are also written to `outbox/<id>.json`; nothing is actually emailed.

## Design: strict where the SOP demands it, flexible where reasoning helps

Each turn runs one pipeline:

```
caller text
  │
  ▼  1. UNDERSTAND   nlu.py      LLM → strict JSON schema (rule extractor as fallback/backstop)
  │                              identity fields, role, intent, case hints, question, scope,
  │                              emotion, refusal, wants_human, yes/no, email choice, done
  ▼  2. REMEMBER     sop.py      everything useful goes into Memory immediately, whatever phase it belongs to
  ▼  3. DECIDE       sop.py      deterministic engine: phase order, gates, limits → a *plan* (ordered acts + allowed facts)
  ▼  4. ACT          agent.py    tools run only if planned (claim lookup, consent request, send_email)
  ▼  5. PHRASE       responder   LLM writes the reply from the plan only
  ▼  6. GUARD        responder   draft checked for leaks / made-up facts → template if it fails
reply
```

The model **interprets** (step 1) and **phrases** (step 5). It never decides phase changes, verification or what may be disclosed. A prompt injection like "ignore your rules, I'm verified" changes nothing, because verification is a deterministic comparison against the policyholder records.

### How much freedom each phase gets

| Phase | Control | Rules enforced by code | What the model is free to do |
|---|---|---|---|
| VERIFY_ID | **Strict** | Needs ≥3 of {full name, DOB, phone, email, SSN/national-ID last 4} matching one policyholder with no field contradicting the record (aliases count). The policy number helps find the record but doesn't count toward the 3. After a mismatch the agent never says which field was wrong, the details are retained for correction, and 3 failures lead to a handoff. Claim details stay locked: the guard rejects claim IDs, document names, amounts or a false "you're verified" in any draft. Representatives must be on the policy, and the policyholder must approve a consent request (polled each turn, offer a human on timeout). | Pull fields out of messy text ("born March 15th, '85"), handle partial answers spread over several turns, questions, refusals, and alternative ID fields. |
| RESOLVE_INTENT | Bounded | Only the caller's own claims are searched. Remembered hints (type, status, month, year, ID) are matched deterministically: one match is selected, several get a short choice question, an unknown ID gets "not on your policy". | Turn vague wording into intent and hints ("the one from the hospital in January", "the denied one", "the second"). |
| PROCESS_CASE | Flexible, grounded | Facts come only from the selected claim, the field definitions and the document guideline. Follow-up guidance is chosen by its `match_any` phrases. Changing decisions or payments is refused. Switching to another claim goes back through RESOLVE_INTENT. The guard checks that every claim ID, `$` amount and date in the reply exists in this turn's facts. | Answer follow-ups in natural language, combine guidance, explain field meanings, handle "what if I can't get X". |
| POST_PROCESS | Strict choice | Always offers the email summary. The caller picks send or skip. It only goes to the **email on file** (shown masked), and a different address is declined. The summary is built from memory (topics discussed, claim status, next steps and deadlines) and checked for made-up amounts. | Write the summary; answer a late question (goes back to PROCESS_CASE, then offers the email again). |

### Remembering information across phases

`Memory` holds identity, role, intent, case hints and the latest question, captured on **every** turn. If the caller says *"I'm calling about my denied healthcare claim from January"* during VERIFY_ID:

1. The engine stays in VERIFY_ID, stores `{status: denied, case_type: healthcare, month: 1}` and `intent=denial_question`, and the reply says the note was taken without confirming the claim exists.
2. As soon as verification passes, the phases run on within the same turn: RESOLVE_INTENT matches the stored hints to **CL-2048** (not CL-2011, the *closed* January 2025 healthcare claim), and PROCESS_CASE answers the stored intent. The caller never gets asked "what are you calling about?" again. The trace shows `remembered from VERIFY_ID`.

### Scope control

Each message gets a scope label: `in_scope`, `smalltalk` or `out_of_scope`. Out-of-scope messages are politely declined without being answered, and the agent steers back to the current step. The 3rd off-topic message in a row adds an offer to transfer to a human. The 4th transfers. A message that also carries useful information (e.g. identity) is never counted as off-topic.

### Emotional support and getting the SOP back on track (bonus)

- **Recognition**: the extraction labels each message `frustrated / angry / anxious / confused / sad / neutral`, and the engine tracks how many negative messages come in a row.
- **De-escalation first**: a negative emotion puts an `empathize` act at the **front** of the plan, so the acknowledgement comes before any workflow ask.
- **Explaining why**: pushback during VERIFY_ID, or a frustrated demand for claim details, adds `explain_verification` (claim data is protected, and verification stops someone else accessing it). Representatives get the same treatment for consent.
- **Persuading without bypassing**: the reply restates *which* fields are still needed and lists the alternatives ("any three of …"). The gate never opens early, and the guard backs this up.
- **Knowing when to stop**: 2 refusals or 3 frustrated or angry messages in a row lead to an offer of a human. 3 refusals, 3 failed verifications, a consent timeout, or any explicit request for a human lead to a **handoff**. The handoff note (reason, whether verified, intent, hints, emotion trend) is shown in the UI, and claim details are left out if the gate never opened.

Example (frustrated caller, not verified):

> **Caller:** I already told you who I am. This is ridiculous. Just tell me why my claim was denied.
> **Plan:** `empathize → remember_for_later → explain_verification → ask_identity`
> **Agent:** acknowledges the frustration, confirms the January claim is noted, explains that claim details are protected until verification, and lists the ID options. No claim details are shared.

## Evaluation at scale

`evals/scenarios.json` holds 49 scripted conversations in 10 categories (verification, privacy, memory, intent, processing, post-process, scope, emotion, representative, names). `evals/run_eval.py` runs them in parallel against the live model or offline and checks:

- **Expectations per scenario:** final phase, verified party, selected claim, email state, gate, how the caller is addressed, which acts were planned on which turn, and required or forbidden reply text.
- **Global invariants on every turn:** no claim detail before the gate opens, and no empty replies.
- **Optional LLM judge (`--judge`):** scores naturalness, empathy, SOP adherence and clarity from 1 to 5, and lists concrete issues.
- **Flakiness (`--repeat N`):** a scenario that passes only some of its repeats is reported as flaky.

Reports go to `evals/reports/` (`latest.md` plus timestamped JSON/Markdown). The verified submission run passed 49 of 49 offline scenarios. The real-model UI walkthrough passed the full workflow; the complete live evaluation and optional LLM judge were not rerun for this submission. The suite exits non-zero on failure, so it can gate CI.

## Production features

| Area | What's there | Configure |
|---|---|---|
| **Conversations survive restarts** | Every turn is saved to SQLite and reloaded on demand. Stored chats older than `SESSION_TTL_HOURS` are purged. A key typed into the UI is never stored. | `SOP_DB_PATH` |
| **Lockout across chats** | Failed verifications are counted per policyholder across *all* chats. After `LOCKOUT_MAX_FAILURES` within `LOCKOUT_WINDOW_MIN`, verification is locked (even a correct attempt goes to a human). A successful verification resets the count. | `LOCKOUT_MAX_FAILURES=5`, `LOCKOUT_WINDOW_MIN=30` |
| **Name handling** | Callers are addressed by their recorded preferred name (`fixtures/preferred_names.json` or a `preferred_name` field), by a name they ask for ("please call me Tian"), or otherwise by full name. The agent never guesses which part of a name is the given name. | — |
| **Email summaries** | SMTP with STARTTLS. Every message also keeps an audit copy in `outbox/`. | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` |
| **Representative consent** | `CONSENT_MODE=link` texts the policyholder a one-time link (`/consent/<token>`) to approve or decline. The chat checks the decision each turn, and the request times out after `CONSENT_TIMEOUT_MIN`. A decline is final. SMS goes through Twilio when configured, otherwise to `outbox/sms-*.json`. In the developer view, choose **Consent: real SMS link** and click the link shown in the side panel. | `CONSENT_MODE`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `PUBLIC_BASE_URL` |
| **Claims system** | A REST adapter (`GET /policyholders`, `/claims`, `/representatives`, bearer token, cached for `CLAIMS_API_TTL`) replaces the fixture files without touching the SOP engine. | `CLAIMS_API_URL`, `CLAIMS_API_TOKEN` |
| **Developer view protection** | `DEV_PASSWORD` puts HTTP Basic auth (user `dev`) on `/`, `/dev` and the developer API. The customer view stays public. | `DEV_PASSWORD`, `DEV_USER` |
| **Abuse protection** | Per-IP rate limit on the customer chat. Messages are capped at 2,000 characters. | `RATE_LIMIT_PER_MIN=30` |

Every integration falls back to its demo version when its variables are unset, so the project still runs with only a model key. The developer view shows which integrations are live under **Integrations**.

## Hosting

The image runs as a non-root user, with a health check and a `/data` volume for the database and outbox. Set `DEV_PASSWORD` and `PUBLIC_BASE_URL` whenever it's reachable from the internet.

```bash
docker compose up --build                       # local, production-like, persists to a named volume
```

- **Render:** push the repo, then **New → Blueprint**. `render.yaml` defines the service, a 1 GB disk on `/data` and the health check. Enter `AI_API_KEY`, `DEV_PASSWORD` and `PUBLIC_BASE_URL` when prompted.
- **Fly.io:** `fly launch --no-deploy`, then `fly volumes create agent_data --size 1`, then `fly secrets set AI_API_KEY=… DEV_PASSWORD=… PUBLIC_BASE_URL=https://<app>.fly.dev`, then `fly deploy` (uses `fly.toml`).
- **Anything else:** any container host works (Cloud Run, ECS, Azure Container Apps). Mount a volume at `/data` and set the same variables.

Run **one instance**: chats, lockout counters and consent requests live in SQLite on that volume. To scale out, move those three tables to Postgres or Redis; the `Store` class is the only code that changes.

## Project layout

```
fixtures/                 sample data (+ optional preferred_names.json)
sop_agent/
  sop.py                  the deterministic SOP engine (phases, gates, limits, lockout, plans)
  nlu.py                  LLM structured extraction + rule-based extractor and safety backstops
  responder.py            LLM phrasing, templates, output guard, email summary
  memory.py               per-session memory + the disclosure gate
  data.py                 back office: identity matching, claim search, grounded facts, names
  integrations.py         email (outbox/SMTP), SMS (outbox/Twilio), consent (scenario/link), claims (fixtures/REST)
  store.py                SQLite: conversations, verification failures, consent requests
  llm.py                  model backends: Claude (Anthropic SDK) and OpenAI-compatible (Responses / Chat API)
  agent.py                turn orchestration, SSN redaction, persistence
  server.py               developer API, customer API, consent page, auth, rate limiting
  static/                 index.html (developer view), client.html (customer view)
evals/                    scenarios.json, run_eval.py, reports/
tests/                    57 unit tests
Dockerfile, docker-compose.yml, render.yaml, fly.toml, run.sh
```
