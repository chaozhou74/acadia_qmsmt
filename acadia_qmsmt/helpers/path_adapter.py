from typing import Literal
import platform
import subprocess
import re


# todo: handle monted drive (labshare)

def detect_platform() -> Literal["windows", "wsl", "linux"]:
    """
    Detects the current platform: 'windows', 'wsl', or 'linux'.
    """
    system = platform.system().lower()
    if "windows" in system:
        return "windows"
    elif "linux" in system:
        if "microsoft" in platform.uname().release.lower():
            return "wsl"
        return "linux"
    raise RuntimeError(f"Unknown platform: {system}")


def is_windows_path(path: str) -> bool:
    """
    Heuristically checks if a given path is a Windows-style path.
    """
    return bool(re.match(r"^[a-zA-Z]:\\", path)) or path.startswith("\\\\")  # UNC or C:\ format


def is_posix_path(path: str) -> bool:
    """
    Heuristically checks if a path is a POSIX (Linux/WSL) path.
    """
    return path.startswith("/") or path.startswith("~")


def to_windows_path(path: str) -> str:
    """
    Converts a Linux/WSL path to a Windows path (only if needed).
    Uses `wslpath -w`.
    """
    if is_windows_path(path):
        return path  # Already a Windows path
    try:
        return subprocess.check_output(["wslpath", "-w", path], text=True).strip()
    except Exception as e:
        print("to_windows_path failed:", e)
        return path


def to_wsl_path(path: str) -> str:
    """
    Converts a Windows path to a Linux/WSL path (only if needed).
    Uses `wslpath -u`.
    """
    if is_posix_path(path):
        return path  # Already a POSIX path
    try:
        return subprocess.check_output(["wslpath", "-u", path], text=True).strip()
    except Exception as e:
        print("to_wsl_path failed:", e)
        return path


def to_local_path(path: str) -> str:
    """
    Converts a given path into a path that makes sense for the current platform:
    - On WSL: converts from Windows to WSL path
    - On Windows: converts from WSL/Linux to Windows path
    - On Linux: returns unchanged
    """
    platform_type = detect_platform()
    if platform_type == "wsl":
        return to_wsl_path(path)
    elif platform_type == "windows":
        return to_windows_path(path)
    return path
