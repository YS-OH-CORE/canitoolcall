"""Warm the Hugging Face cache with every tokenizer/config/template file the fixtures pin.

Downloads, at the pinned revisions, the files adapters read (never weights) for
every ``family.json`` reference model, its declared mirror, and every fixture
``tokenizer`` pin. Afterwards replays can run with ``HF_HUB_OFFLINE=1`` (DESIGN.md
adapter rule 10), so bulk runs never hit Hub rate limits. Repository code
(``tokenization_*.py``) is fetched only for the reviewed pins in
``canitoolcall.adapters.base.REMOTE_CODE_ALLOWLIST``. Set ``HF_TOKEN`` to raise
anonymous rate limits; it is optional because every pinned repo is public.

Run from the repo root::

    uv run --no-project --with huggingface-hub==1.33.0 python scripts/engines/prefetch_hf.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from canitoolcall.adapters.base import trusts_remote_code  # noqa: E402

ALLOW = [
    "config.json",
    "generation_config.json",
    "params.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "vocab.json",
    "merges.txt",
    "tekken.json",
    "chat_template.jinja",
    "chat_template.json",
    "*.jinja",
    "encoding/*.py",
]
REMOTE_CODE = ["tokenization_*.py", "configuration_*.py"]
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def pins(root: Path = ROOT) -> dict[tuple[str, str], None]:
    out: dict[tuple[str, str], None] = {}
    for fam in sorted((root / "fixtures").glob("*/family.json")):
        for ref in json.loads(fam.read_text(encoding="utf-8"))["reference_models"]:
            out[(ref["repo"], ref["revision"])] = None
            if ref.get("mirror"):
                out[(ref["mirror"]["repo"], ref["mirror"]["revision"])] = None
    for jl in sorted((root / "fixtures").glob("*/*.jsonl")):
        for line in jl.read_text(encoding="utf-8").splitlines():
            tok = json.loads(line).get("tokenizer") if line.strip() else None
            if tok:
                out[(tok["repo"], tok["revision"])] = None
    return out


def main() -> int:
    from huggingface_hub import snapshot_download

    failed = 0
    for repo, revision in pins():
        if not HEX40.match(revision):
            print(f"skip {repo}@{revision}: not a full commit sha", file=sys.stderr)
            failed += 1
            continue
        if repo.startswith("meta-llama/"):
            continue  # gated; fixtures pin the documented ungated mirror instead
        allow = ALLOW + (REMOTE_CODE if trusts_remote_code(repo, revision) else [])
        try:
            snapshot_download(repo, revision=revision, allow_patterns=allow)
            print(f"cached {repo}@{revision[:12]}")
        except Exception as e:
            print(f"FAILED {repo}@{revision[:12]}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
