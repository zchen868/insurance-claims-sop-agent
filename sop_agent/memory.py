"""Per-session working memory. Anything the caller says is stored here as soon
as it is heard, even when it belongs to a later phase (e.g. a case hint given
during VERIFY_ID). The SOP engine reads it when the phase is reached."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

PHASES = ("VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS", "ENDED", "HANDOFF")


@dataclass
class Memory:
    phase: str = "VERIFY_ID"

    # VERIFY_ID
    caller_role: str = "unknown"            # policyholder | representative | unknown
    rep_name: str | None = None
    rep_relationship: str | None = None
    identity: dict = field(default_factory=dict)   # field -> value as given
    verified_party_id: str | None = None
    verified_fields: list = field(default_factory=list)
    verify_failures: int = 0
    preferred_name: str | None = None       # "call me ..." from the caller
    address_as: str | None = None           # name the agent uses once verified
    last_failed_identity: dict = field(default_factory=dict)
    refusals: int = 0

    # representative consent
    consent_scenario: str = "default"
    consent_status: str | None = None       # None | pending | approved | timeout
    consent_polls: int = 0
    consent_ref: str | None = None          # token of a link-based consent request
    consent_link: str | None = None

    # RESOLVE_INTENT (may be filled during any phase)
    intent: str | None = None
    case_hints: dict = field(default_factory=dict)
    pending_question: str | None = None
    selected_case_id: str | None = None
    candidate_case_ids: list = field(default_factory=list)
    claims_listed: bool = False

    # PROCESS_CASE
    discussed: list = field(default_factory=list)  # [{case_id, question, topics}]
    cases_discussed: list = field(default_factory=list)

    # POST_PROCESS
    email_state: str = "not_offered"        # not_offered | offered | sent | skipped
    email_record: dict | None = None
    email_requested: bool = False

    # conversation control
    off_topic_streak: int = 0
    negative_streak: int = 0
    emotions: list = field(default_factory=list)
    human_offered: bool = False
    handoff_reason: str | None = None
    handoff_note: str | None = None
    last_agent_question: str | None = None  # what we last asked (for yes/no)

    @property
    def gate_open(self) -> bool:
        """Claim details may be disclosed only when identity is verified and,
        for a representative, the policyholder has approved consent."""
        if not self.verified_party_id:
            return False
        if self.caller_role == "representative":
            return self.consent_status == "approved"
        return True

    def to_dict(self):
        d = asdict(self)
        d["gate_open"] = self.gate_open
        return d
