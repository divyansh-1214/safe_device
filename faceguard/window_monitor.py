# faceguard/window_monitor.py
# Reads the active window title and process name using platform-native APIs.
# Supports Windows (ctypes), Linux (xdotool), macOS (AppKit).

import platform

PLATFORM = platform.system()   # 'Windows', 'Linux', 'Darwin'


def get_active_window() -> dict:
    """
    Returns dict with title and process name of the active window.
    """
    try:
        if PLATFORM == "Windows":
            return _get_windows()
        elif PLATFORM == "Linux":
            return _get_linux()
        elif PLATFORM == "Darwin":
            return _get_mac()
    except Exception:
        pass
    return {"title": "", "process": ""}


def _get_windows() -> dict:
    import ctypes
    import ctypes.wintypes

    hwnd = ctypes.windll.user32.GetForegroundWindow()

    # Get window title
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
    title = buf.value

    # Get process name from PID
    import psutil
    pid = ctypes.wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    try:
        process = psutil.Process(pid.value).name()
    except Exception:
        process = ""

    return {"title": title, "process": process}


def _get_linux() -> dict:
    import subprocess

    try:
        win_id = subprocess.check_output(
            ['xdotool', 'getwindowfocus'], text=True
        ).strip()

        title = subprocess.check_output(
            ['xdotool', 'getwindowname', win_id], text=True
        ).strip()

        pid = subprocess.check_output(
            ['xdotool', 'getwindowpid', win_id], text=True
        ).strip()

        import psutil
        process = psutil.Process(int(pid)).name()
    except Exception:
        title, process = "", ""

    return {"title": title, "process": process}


def _get_mac() -> dict:
    from AppKit import NSWorkspace  # type: ignore[import-not-found]
    active_app = NSWorkspace.sharedWorkspace().activeApplication()
    return {
        "title": active_app.get('NSApplicationName', ''),
        "process": active_app.get('NSApplicationBundleIdentifier', '')
    }


def minimize_active_window():
    """Minimizes the current foreground window (platform-native)."""
    try:
        if PLATFORM == "Windows":
            import ctypes
            SW_MINIMIZE = 6
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)

        elif PLATFORM == "Linux":
            import subprocess
            subprocess.run(['xdotool', 'getwindowfocus', 'windowminimize'])

        elif PLATFORM == "Darwin":
            import subprocess
            subprocess.run([
                'osascript', '-e',
                'tell application "System Events" to keystroke "m" '
                'using command down'
            ])
    except Exception as e:
        print(f"[Monitor] Could not minimize window: {e}")
