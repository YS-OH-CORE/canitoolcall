"""Refuse to release while pre-publication placeholders remain.

Run from the repo root before tagging (the release workflow runs it too)::

    python scripts/check_release.py

Checks that the repository owner, the Code of Conduct contact and the security
policy are filled in, and that ``[project.urls]`` names the real repository.
Exit code 0 when ready, 1 otherwise (each problem is printed).
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the release workflow runs 3.12
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDERS = {
    "<owner>": ("README.md", "pyproject.toml", "CONTRIBUTING.md", "SECURITY.md"),
    "CONTACT METHOD": ("CODE_OF_CONDUCT.md",),
    "This is a stub": ("SECURITY.md",),
}


def problems(root: Path = ROOT) -> list[str]:
    out: list[str] = []
    for needle, files in PLACEHOLDERS.items():
        for name in files:
            path = root / name
            if not path.is_file():
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if needle in line:
                    out.append(f"{name}:{n}: placeholder {needle!r} is not filled in")
    if tomllib is not None:
        meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        urls = meta.get("project", {}).get("urls", {})
        for key in ("Homepage", "Issues"):
            if not urls.get(key):
                out.append(f"pyproject.toml: [project.urls] {key} is missing (set it to the real repository)")
    return out


def main() -> int:
    found = problems()
    for p in found:
        print(p, file=sys.stderr)
    if found:
        print(f"{len(found)} release blocker(s); see docs/PUBLISHING.md section 1", file=sys.stderr)
        return 1
    print("no release placeholders left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
