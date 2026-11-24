#!/usr/bin/env python3
"""
Automated Public Release, Privacy (PII), and Clean-Room Audit Gate.

Checks every tracked/non-ignored file (and optionally git commit metadata) for:
  1. Hardcoded local OS home directory paths (/Users/<name>, /home/<name>, C:\\Users\\<name>)
  2. Personal email addresses (allows only GitHub noreply or RFC example.com domains)
  3. Secrets, API keys, private key headers, or credential patterns
  4. Binary-extraction / disassembly markers or forbidden proprietary extensions
  5. Optional local-only extra denylist terms via AUDIT_EXTRA_TERMS env var (comma-separated)
     or --extra-terms-file (never commit local denylist files to git!)

Exit code 0 = clean; Exit code 1 = violations found.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git",
    "target",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "captures",
}

FORBIDDEN_EXTENSIONS = {
    ".dylib",
    ".so",
    ".dll",
    ".exe",
    ".component",
    ".vst3",
    ".clap",
    ".pkg",
    ".dmg",
    ".pem",
    ".key",
    ".p12",
    ".nam",
    ".wav",
    ".npz",
}

# Structural regexes (zero literal PII embedded in this public file)
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "Hardcoded macOS home path",
        re.compile(r"/Users/(?!<|\.\.\.|[a-z]*user|you|example)[A-Za-z0-9._-]{2,}"),
    ),
    (
        "Hardcoded Linux home path",
        re.compile(r"/home/(?!<|\.\.\.|[a-z]*user|you|example|runner)[A-Za-z0-9._-]{2,}"),
    ),
    (
        "Hardcoded Windows home path",
        re.compile(r"[A-Za-z]:\\Users\\(?!<|\.\.\.|[a-z]*user|you|example|runneradmin)[A-Za-z0-9._-]{2,}"),
    ),
    (
        "Non-noreply email address",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com\b|example\.com\b|example\.org\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    (
        "Private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "Potential API key or token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
    ),
    (
        "Binary extraction marker (not clean-room)",
        re.compile(r"__TEXT,__const|__DATA,__const|Extracted from binary|kAudioUnitProperty_ClassInfo_Dump"),
    ),
]


def is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(4096)
        return b"\x00" in chunk
    except OSError:
        return True


def iter_repo_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            files.append(Path(dirpath) / fname)
    return sorted(files)


def load_extra_terms(extra_file: str | None) -> list[str]:
    terms: list[str] = []
    env_val = os.environ.get("AUDIT_EXTRA_TERMS", "").strip()
    if env_val:
        terms.extend(t.strip() for t in env_val.split(",") if t.strip())
    if extra_file:
        p = Path(extra_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
    return terms


def audit_files(root: Path, extra_terms: list[str]) -> list[str]:
    errors: list[str] = []
    self_path = Path(__file__).resolve()
    extra_regexes = [(f"Extra denylist term ({t})", re.compile(re.escape(t), re.IGNORECASE)) for t in extra_terms]

    for fpath in iter_repo_files(root):
        rel = fpath.relative_to(root)
        ext = fpath.suffix.lower()
        if ext in FORBIDDEN_EXTENSIONS:
            errors.append(f"[FORBIDDEN EXT] {rel}: disallowed extension '{ext}' for public clean-room repo")
            continue

        if is_binary_file(fpath):
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"[READ ERR] {rel}: {exc}")
            continue

        active_patterns = list(extra_regexes)
        # Skip structural regex self-matches inside this audit script itself
        if fpath.resolve() != self_path:
            active_patterns.extend(PATTERNS)

        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, rx in active_patterns:
                m = rx.search(line)
                if m:
                    errors.append(f"[{label}] {rel}:{lineno}: matched '{m.group(0)}'")
    return errors


def audit_git_log(root: Path, extra_terms: list[str]) -> list[str]:
    errors: list[str] = []
    if not (root / ".git").exists():
        return errors
    proc = subprocess.run(
        ["git", "-C", str(root), "log", "--all", "--format=%H %an <%ae> | %cn <%ce> | %s"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return errors

    email_rx = re.compile(
        r"<([A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com>|example\.com>)[A-Za-z0-9.-]+\.[A-Za-z]{2,})>"
    )
    extra_regexes = [(t, re.compile(re.escape(t), re.IGNORECASE)) for t in extra_terms]

    for line in proc.stdout.splitlines():
        m = email_rx.search(line)
        if m:
            errors.append(f"[GIT LOG EMAIL] Non-noreply commit email found: {line}")
        for term, rx in extra_regexes:
            if rx.search(line):
                errors.append(f"[GIT LOG EXTRA TERM '{term}'] {line}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean-room & privacy audit gate")
    parser.add_argument("--check-git-log", action="store_true", help="Also audit git commit authors and messages")
    parser.add_argument("--extra-terms-file", default=None, help="Optional local file of extra forbidden strings")
    args = parser.parse_args()

    extra_terms = load_extra_terms(args.extra_terms_file)
    errors = audit_files(REPO_ROOT, extra_terms)
    if args.check_git_log:
        errors.extend(audit_git_log(REPO_ROOT, extra_terms))

    if errors:
        print(f"AUDIT FAILED ({len(errors)} finding(s)):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("AUDIT PASSED: 0 privacy, path, secret, or clean-room violations found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
