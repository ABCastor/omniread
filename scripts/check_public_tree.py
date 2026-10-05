"""Fail on private paths and addresses in tracked public text."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+(?:/|\b)"),
    re.compile(r"[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)
ALLOWED_DOMAINS = {"example.com", "example.org", "example.net", "example.test", "test.invalid"}
failures = []
files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
for name in filter(None, files):
    try:
        content = (ROOT / name).read_text()
    except (UnicodeDecodeError, IsADirectoryError):
        continue
    for line_no, line in enumerate(content.splitlines(), 1):
        for pattern in PATTERNS:
            for match in pattern.finditer(line):
                value = match.group()
                if "@" in value and value.rsplit("@", 1)[-1] in ALLOWED_DOMAINS:
                    continue
                failures.append(f"{name}:{line_no}: private path or address")
if failures:
    print("\n".join(failures))
    sys.exit(1)
print("Public tree: no private home paths or non-placeholder email addresses")
