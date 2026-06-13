# faceguard/access_rules.py
# Checks the active window title/process against the user's profile
# from profiles.json.  Blocked list takes priority over allowed list.

import json
import os
import datetime


def load_profiles(path: str = None) -> dict:
    """Load all user profiles from profiles.json.

    Default path resolves to `<project_root>/profiles.json`.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "profiles.json",
        )
    with open(path, "r") as f:
        return json.load(f)


def check_access(window: dict, profile: dict) -> tuple:
    """Determine whether the active window is allowed for this user.

    Args:
        window:  {"title": str, "process": str}  — from window_monitor
        profile: one user's entry from profiles.json

    Returns:
        (allowed: bool, reason: str)
    """
    title   = window.get("title", "").lower()
    process = window.get("process", "").lower()

    # ── Admin bypasses everything ─────────────────────────────────────────
    if profile.get("role") == "admin":
        return True, "admin access"

    # ── Time restriction ──────────────────────────────────────────────────
    now   = datetime.datetime.now().hour
    start = profile.get("allowed_hours", {}).get("start", 0)
    end   = profile.get("allowed_hours", {}).get("end", 23)

    if not (start <= now <= end):
        return False, f"Access not allowed outside {start}:00 - {end}:00"

    # ── Blocked list (takes priority) ─────────────────────────────────────
    for blocked in profile.get("blocked_urls", []):
        if blocked.lower() in title or blocked.lower() in process:
            return False, f"'{blocked}' is blocked for your profile"

    # ── Allowed list ──────────────────────────────────────────────────────
    allowed_urls = profile.get("allowed_urls", [])

    if "*" in allowed_urls:
        return True, "all urls allowed"

    for allowed in allowed_urls:
        if allowed.lower() in title or allowed.lower() in process:
            return True, f"'{allowed}' is allowed"

    # ── Default: block if not explicitly allowed ──────────────────────────
    return False, "not in your allowed list"
