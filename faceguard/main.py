"""
faceguard/main.py — FaceGuard Desktop Access Control (Module 8)
================================================================
Entry point for the background access-control agent.

Run with:
    python -m faceguard.main

Requires FastAPI backend (`api.py`) to be running first:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Flow:
    1. Camera → capture frame → POST /verify
    2. Verified → load profile from profiles.json
    3. Monitor loop: poll active window every 1.5 s
    4. Blocked keyword found → minimize window + desktop notification
    5. Session expires after session_minutes → re-verification required
"""

import time

from faceguard.window_monitor import get_active_window, minimize_active_window
from faceguard.access_rules import load_profiles, check_access
from faceguard.notifier import show_warning
from faceguard.face_verify import capture_and_verify

MONITOR_INTERVAL = 1.5       # seconds between each window check
SESSION_WARN_AHEAD = 5       # warn N minutes before session expires

# Windows whose title contains any of these keywords are never blocked,
# to avoid locking the user out of essential system tools.
SKIP_KEYWORDS = [
    "faceguard", "task manager", "cmd", "terminal",
    "powershell", "python", "settings", "control panel",
]


def run():
    print("=" * 50)
    print("  FaceGuard Desktop Access Control")
    print("=" * 50)

    profiles = load_profiles()

    # ── Step 1: Face verification ─────────────────────────────────────────
    print("\n[Step 1] Scanning your face...")
    result = capture_and_verify()

    if not result:
        print("[FaceGuard] Could not verify face — exiting")
        return

    name = result["name"]
    profile = profiles.get(name)

    if not profile:
        print(f"[FaceGuard] No profile found for '{name}'")
        return

    # Session timer
    session_minutes = profile.get("session_minutes", 30)
    session_expiry = time.time() + (session_minutes * 60)

    print(f"\n[FaceGuard] Welcome, {name}!")
    print(f"  Role    : {profile['role']}")
    print(f"  Session : {session_minutes} minutes")
    print(f"  Blocked : {profile.get('blocked_urls', [])}")
    print(f"  Allowed : {profile.get('allowed_urls', ['*'])}")
    print("\n[FaceGuard] Monitoring started. Press Ctrl+C to stop.\n")

    # ── Step 2: Monitor loop ──────────────────────────────────────────────
    last_blocked_title = ""

    while True:
        try:
            # ── Session expiry check ──────────────────────────────────────
            remaining = session_expiry - time.time()

            if remaining <= 0:
                print(f"\n[FaceGuard] Session expired for {name}")
                show_warning(name, "Session expired — please verify again", "")
                break

            if remaining <= SESSION_WARN_AHEAD * 60:
                mins_left = int(remaining // 60)
                print(f"[FaceGuard] Warning — {mins_left} min left in session")

            # ── Read active window ────────────────────────────────────────
            window = get_active_window()
            title = window.get("title", "")

            if not title:
                time.sleep(MONITOR_INTERVAL)
                continue

            # Skip internal / system windows
            if any(kw in title.lower() for kw in SKIP_KEYWORDS):
                time.sleep(MONITOR_INTERVAL)
                continue

            # ── Access check ──────────────────────────────────────────────
            allowed, reason = check_access(window, profile)

            if not allowed:
                # Only act on a *new* blocked window to avoid repeated minimize
                if title != last_blocked_title:
                    print(f"[BLOCKED] {title}")
                    print(f"          Reason: {reason}")
                    minimize_active_window()
                    show_warning(name, reason, title)
                    last_blocked_title = title
            else:
                last_blocked_title = ""   # reset when an allowed window is active

            time.sleep(MONITOR_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n[FaceGuard] Stopped by user")
            break

    print("[FaceGuard] Session ended")


if __name__ == "__main__":
    run()
