# faceguard/notifier.py
# Desktop warning popup using plyer.
# Runs notifications in a daemon thread so the monitor loop is never blocked.
# Has a cooldown to prevent notification spam on rapid window switching.

import threading
import time

try:
    from plyer import notification
except ImportError:
    notification = None
    print("[Notifier] plyer not installed — notifications disabled. "
          "Install with: pip install plyer")

_last_notification_time: float = 0
COOLDOWN_SECONDS: int = 5


def show_warning(name: str, reason: str, window_title: str):
    """Show a desktop notification warning.

    Args:
        name:         verified user's name
        reason:       why the window was blocked
        window_title: the blocked window's title (truncated in the popup)
    """
    global _last_notification_time

    if notification is None:
        return

    now = time.time()
    if now - _last_notification_time < COOLDOWN_SECONDS:
        return   # cooldown active — skip

    _last_notification_time = now

    def _notify():
        try:
            notification.notify(
                title="FaceGuard — Access Blocked",
                message=f"{name}: {reason}\n\n{window_title[:60]}",
                app_name="FaceGuard",
                timeout=4,
            )
        except Exception as e:
            print(f"[Notifier] Failed to show notification: {e}")

    threading.Thread(target=_notify, daemon=True).start()
