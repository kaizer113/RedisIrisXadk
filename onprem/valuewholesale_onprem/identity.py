"""Server-issued synthetic-persona sessions, separate from production authentication."""

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from uuid import uuid4

COOKIE = "valuewholesale_ga_session"
SESSION_SECONDS = 86400


@dataclass(frozen=True)
class Identity:
    member_id: str
    session_id: str

    @property
    def scope(self) -> str:
        return f"valuewholesale-ga:{self.member_id}:{self.session_id}"

    @property
    def owner(self) -> str:
        return f"valuewholesale-onprem:{self.member_id}"


def issue(member_id: str, secret: str) -> tuple[Identity, str]:
    identity = Identity(member_id, uuid4().hex)
    data = json.dumps(
        {
            "member": member_id,
            "session": identity.session_id,
            "expires": int(time.time()) + SESSION_SECONDS,
        },
        separators=(",", ":"),
    )
    payload = base64.urlsafe_b64encode(data.encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return identity, payload + "." + signature


def verify(cookie: str, secret: str) -> Identity:
    if len(secret) < 32:
        raise ValueError("Configure ONPREM_SESSION_SECRET with at least 32 random characters")
    try:
        payload, signature = cookie.split(".")
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid session")
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if data["expires"] <= time.time():
            raise ValueError("Session expired")
        return Identity(data["member"], data["session"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Choose a demo member to start a session") from exc
