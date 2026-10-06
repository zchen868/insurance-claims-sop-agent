"""Thin Claude wrapper. The model is used for two bounded jobs only:
understanding the caller (structured JSON) and phrasing the reply the SOP
engine has already decided on. It never decides phase transitions."""
from __future__ import annotations

import json
import logging
import os

import anthropic
import httpx

log = logging.getLogger("sop.llm")

MODEL = os.environ.get("SOP_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("SOP_EFFORT", "low")  # chat route: low effort keeps latency down


class LLM:
    def __init__(self, api_key: str | None = None):
        # Explicit key from the UI wins; otherwise the SDK resolves
        # ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.use_fallbacks = True
        self.last_error: str | None = None

    def _create(self, **kw):
        if self.use_fallbacks:
            try:
                return self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            except anthropic.BadRequestError as e:
                if "fallback" not in str(e).lower():
                    raise
                log.warning("server-side fallbacks rejected, continuing without: %s", e)
                self.use_fallbacks = False
        return self.client.messages.create(**kw)

    def _call(self, system, messages, max_tokens, output_config):
        try:
            resp = self._create(model=MODEL, max_tokens=max_tokens, system=system,
                                messages=messages, output_config=output_config)
        except anthropic.AuthenticationError:
            self.last_error = "Invalid API key"
            return None
        except anthropic.RateLimitError:
            self.last_error = "Rate limited"
            return None
        except anthropic.APIStatusError as e:
            self.last_error = f"API error {e.status_code}: {e.message}"
            return None
        except anthropic.APIConnectionError:
            self.last_error = "Network error"
            return None
        if resp.stop_reason == "refusal":
            self.last_error = "Model declined"
            return None
        self.last_error = None
        return next((b.text for b in resp.content if b.type == "text"), None)

    def json(self, system, user, schema, max_tokens=2000):
        text = self._call(system, [{"role": "user", "content": user}], max_tokens,
                          {"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}})
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.last_error = "Bad JSON from model"
            return None

    def text(self, system, user, max_tokens=1500):
        return self._call(system, [{"role": "user", "content": user}], max_tokens, {"effort": EFFORT})


    @property
    def label(self):
        return f"claude ({MODEL})"


class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint (DeepSeek, etc.), using either the
    Responses API (`AI_API_FORMAT=responses`) or Chat Completions (`chat`)."""

    def __init__(self, api_key, base_url, model, api_format="responses", effort="low", provider="openai-compatible"):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=90,
                                 headers={"Authorization": f"Bearer {api_key}"})
        self.model, self.format, self.effort, self.provider = model, api_format, effort, provider
        self.last_error: str | None = None

    @property
    def label(self):
        return f"{self.provider} ({self.model})"

    def _post(self, path, body):
        try:
            r = self.http.post(path, json=body)
        except httpx.HTTPError as e:
            self.last_error = f"Network error: {e}"
            return None
        if r.status_code != 200:
            self.last_error = f"API error {r.status_code}: {r.text[:200]}"
            return None
        self.last_error = None
        return r.json()

    def _call(self, system, user, schema=None, max_tokens=3000):
        if self.format == "responses":
            body = {"model": self.model, "instructions": system, "input": user,
                    "max_output_tokens": max_tokens, "reasoning": {"effort": self.effort}}
            if schema:
                body["text"] = {"format": {"type": "json_schema", "name": "extraction", "strict": True, "schema": schema}}
            d = self._post("/responses", body)
            if d is None:
                return None
            for item in d.get("output", []):
                if item.get("type") == "message":
                    return "".join(c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text")
            self.last_error = "No message in response"
            return None
        body = {"model": self.model, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system + (
                    "\n\nReturn only a JSON object matching this JSON Schema:\n" + json.dumps(schema) if schema else "")},
                    {"role": "user", "content": user}]}
        if schema:
            body["response_format"] = {"type": "json_object"}
        d = self._post("/chat/completions", body)
        if d is None:
            return None
        return d["choices"][0]["message"].get("content")

    def json(self, system, user, schema, max_tokens=3000):
        text = self._call(system, user, schema, max_tokens)
        if text is None:
            return None
        try:
            return json.loads(text.strip().removeprefix("```json").removesuffix("```"))
        except json.JSONDecodeError:
            self.last_error = "Bad JSON from model"
            return None

    def text(self, system, user, max_tokens=3000):
        return self._call(system, user, None, max_tokens)


def build_llm(api_key: str | None):
    """Pick the model backend from the environment. AI_PROVIDER=anthropic (or
    unset with an Anthropic key) uses Claude; any other AI_PROVIDER uses the
    OpenAI-compatible client with AI_BASE_URL / AI_API_KEY / AI_MODEL. A key
    entered in the UI overrides the configured key for that session."""
    provider = os.environ.get("AI_PROVIDER", "").lower()
    try:
        if provider and provider not in ("anthropic", "claude"):
            key = api_key or os.environ.get("AI_API_KEY")
            if key and os.environ.get("AI_BASE_URL") and os.environ.get("AI_MODEL"):
                return OpenAICompatLLM(key, os.environ["AI_BASE_URL"], os.environ["AI_MODEL"],
                                       os.environ.get("AI_API_FORMAT", "responses"),
                                       os.environ.get("SOP_EFFORT", "low"), provider)
            return None
        if api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") \
                or (provider and os.environ.get("AI_API_KEY")):
            return LLM(api_key or (os.environ.get("AI_API_KEY") if provider else None))
    except Exception as e:  # missing credentials etc.
        log.warning("LLM unavailable, running offline: %s", e)
    return None
