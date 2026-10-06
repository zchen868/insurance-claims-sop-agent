"""Shared test environment: offline mode, fixed date, throwaway outbox and database.
Runs before any test module imports the app."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["SOP_OUTBOX_DIR"] = os.path.join(_tmp, "outbox")
os.environ["SOP_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["SOP_TODAY"] = "2026-03-05"
for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AI_PROVIDER", "AI_API_KEY", "SMTP_HOST",
          "TWILIO_ACCOUNT_SID", "CLAIMS_API_URL", "CONSENT_MODE", "DEV_PASSWORD"):
    os.environ.pop(k, None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_lockout_counters():
    """Failed-verification counters are shared across chats by design; isolate tests."""
    from sop_agent.store import get_store
    get_store()._q("DELETE FROM verify_failures")
    from sop_agent import server
    server._hits.clear()        # per-IP rate limit counters
    yield
