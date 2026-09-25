from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone


def build_config_fingerprint(payload: dict) -> str:
    """Return a deterministic short digest for comparison-relevant config."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def build_run_id(payload: dict) -> str:
    """Generate a unique run ID from a timestamp + content hash.

    Uses microsecond-resolution timestamps and mixes in a random nonce
    to prevent collisions when the same config is run in rapid succession.
    The hash includes all scoring-relevant config (model, scenarios,
    temperature, seed, error_rate, backend, extra_params).
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    # Mix in a random nonce to guarantee uniqueness even for identical
    # payloads within the same microsecond (e.g. parallel launchers).
    nonce = os.urandom(4).hex()
    payload_with_nonce = {**payload, "_nonce": nonce}
    digest = build_config_fingerprint(payload_with_nonce)[:8]
    return f"{ts}_{digest}"
