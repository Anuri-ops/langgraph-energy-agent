"""Fail CI if credentials, local vector data or private traces are tracked in Git.

No file contents are printed; the check only reports prohibited relative paths.
Works in a repository; skips Git-specific checks in extracted ZIP builds.
"""
from __future__ import annotations

from pathlib import PurePosixPath
import subprocess


def is_sensitive_path(path: str) -> bool:
    p = PurePosixPath(path.replace("\\", "/"))
    parts = p.parts
    name = p.name.lower()
    if name == ".env.example":
        return False
    if name == ".env" or name.startswith(".env."):
        return True
    if name.endswith((".pem", ".p12", ".pfx", ".key")):
        return True
    if any(part in {"chroma_db", "traces", "__pycache__"} for part in parts):
        return True
    return len(parts) >= 2 and parts[:2] == ("evaluation", "results")


def main() -> None:
    try:
        result = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=False)
    except FileNotFoundError:
        print("Git is not installed: skipped Git index check (ZIP workflow).")
        return
    if result.returncode != 0:
        print("No Git repository available: skipped Git index check (ZIP workflow).")
        return
    paths = [x.decode("utf-8", errors="replace") for x in result.stdout.split(b"\0") if x]
    forbidden = [p for p in paths if is_sensitive_path(p)]
    if forbidden:
        print("BLOCKED: sensitive files are tracked; remove them from Git before publishing:")
        for p in forbidden:
            print("  " + p)
        raise SystemExit(1)
    print("Tracked-file security check passed.")


if __name__ == "__main__":
    main()
