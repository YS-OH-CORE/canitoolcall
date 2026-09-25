#!/usr/bin/env bash
# Make vocab-only GGUFs with llama.cpp's own converter (convert_hf_to_gguf.py
# --vocab-only) at the pinned llama.cpp commit:
#
#   .engines/gguf/<org>--<model>.vocab.gguf     tokenizer + chat template, no tensors
#   .engines/gguf/<org>--<model>.vocab.json     provenance: repo, revision, llama.cpp commit, sha256s
#   .engines/gguf/<org>--<model>.vocab.error.json   written instead when download/conversion fails
#
# The converter runs in .venvs/llamacpp (llama.cpp's pinned
# requirements-convert_hf_to_gguf.txt, i.e. transformers 4.57.6). Repos whose
# tokenizer files need transformers 5 (e.g. Gemma 4, GLM-5.x) are retried in
# .engines/llamacpp/convert-tf5 (same requirements, transformers 5.17.0), and the
# env used is recorded in the .vocab.json sidecar ("converter_env"). Failures the
# converter itself says how to fix (see REMEDIES) are retried with that fix, also
# recorded ("converter_args", "converter_patches").
#
# Only config/tokenizer/template files are downloaded (never weights), into the
# normal HF cache at the pinned revision. The llama.cpp adapter detokenizes with
# these files exactly like llama-server; the Ollama adapter reads them read-only.
#
# Usage:
#   scripts/engines/gguf_vocab.sh --all                    # every model the fixtures pin
#   scripts/engines/gguf_vocab.sh ORG/MODEL@REVISION ...   # explicit repos
#   scripts/engines/gguf_vocab.sh --force ...              # rebuild even if up to date
# Needs scripts/engines/llamacpp.sh to have run (llama.cpp checkout + .venvs/llamacpp).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${CANITOOLCALL_LLAMACPP_PYTHON:-$ROOT/.venvs/llamacpp/bin/python}"
SRC="$ROOT/.engines/llamacpp/llama.cpp"
OUT="$ROOT/.engines/gguf"

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,19p' "$0"; exit 0; }
[[ -x "$PY" ]] || { echo "missing $PY; run scripts/engines/llamacpp.sh first" >&2; exit 1; }
[[ -f "$SRC/convert_hf_to_gguf.py" ]] || { echo "missing $SRC; run scripts/engines/llamacpp.sh first" >&2; exit 1; }
mkdir -p "$OUT"

TF5_PY="$ROOT/.engines/llamacpp/convert-tf5/bin/python"
exec "$PY" - "$ROOT" "$SRC" "$OUT" "$TF5_PY" "$@" <<'PYEOF'
"""Driver for gguf_vocab.sh (runs in .venvs/llamacpp)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

root, src, out, tf5_py = (Path(p) for p in sys.argv[1:5])
args = sys.argv[5:]
force = "--force" in args
args = [a for a in args if a != "--force"]

# Tokenizer, config and template files only: never weights.
ALLOW = [
    "config.json", "generation_config.json", "params.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.model", "*.tiktoken", "tiktoken.model", "vocab.json", "merges.txt", "vocab.txt",
    "tekken.json", "chat_template.jinja", "chat_template.json", "*.jinja",
    "tokenization_*.py", "configuration_*.py",
]


def _set_add_bos_token(model_dir: Path) -> None:
    cfg = model_dir / "tokenizer_config.json"
    data = json.loads(cfg.read_text())
    data["add_bos_token"] = True
    cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# Failures the converter itself tells you how to fix for a vocab-only export:
# (text in stderr, remedy recorded in the sidecar, how to apply it). Each remedy
# leaves the vocab and chat template untouched and is recorded, never hidden.
REMEDIES: list[tuple[str, str, Callable[[Path], None]]] = [
    # Qwen3-Next-style converters count MTP layers from the weight index, which a
    # vocab-only export lacks; --no-mtp only drops the MTP draft tensors.
    ("opt_num_mtp_layers", "--no-mtp", lambda _d: None),
    # conversion/deepseek.py (V3.2): "Change value of add_bos_token to true in
    # tokenizer_config.json file." BOS only affects prompt tokenization.
    ("Change value of add_bos_token to true", "tokenizer_config.json: add_bos_token=true", _set_add_bos_token),
    # Mistral native-format repos (params.json + tekken.json, no HF config.json /
    # tokenizer): llama.cpp converts them with its --mistral-format flag.
    ("config.json'", "--mistral-format", lambda _d: None),
    ("MistralCommonBackend has no attribute vocab", "--mistral-format", lambda _d: None),
    ("pre-tokenizer 'tekken' is not supported", "--mistral-format", lambda _d: None),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wanted_from_fixtures() -> dict[str, str]:
    """repo -> revision, from every family.json reference model and fixture tokenizer pin."""
    pins: dict[str, str] = {}
    fixture_roots = [root / "fixtures", root / "tests" / "core" / "data" / "fixtures"]
    for fr in fixture_roots:
        for fam in sorted(fr.glob("*/family.json")):
            for ref in json.loads(fam.read_text()).get("reference_models", []):
                pins.setdefault(ref["repo"], ref["revision"])
        for jl in sorted(fr.glob("*/*.jsonl")):
            for line in jl.read_text().splitlines():
                if not line.strip():
                    continue
                tok = json.loads(line).get("tokenizer")
                if tok:
                    prev = pins.setdefault(tok["repo"], tok["revision"])
                    if prev != tok["revision"]:
                        print(f"warning: {tok['repo']} pinned at {prev} and {tok['revision']}; using {prev}",
                              file=sys.stderr)
    return pins


def stem(repo: str) -> str:
    return repo.replace("/", "--")


def commit() -> str:
    return subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def build(repo: str, revision: str, llama_commit: str) -> bool:
    gguf = out / f"{stem(repo)}.vocab.gguf"
    meta = out / f"{stem(repo)}.vocab.json"
    err = out / f"{stem(repo)}.vocab.error.json"
    if not force and gguf.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if (m.get("revision"), m.get("llama_cpp_commit"), m.get("sha256")) == (revision, llama_commit, sha256(gguf)):
            print(f"up to date: {gguf.name}")
            return True
    err.unlink(missing_ok=True)

    from huggingface_hub import snapshot_download

    try:
        snap = Path(snapshot_download(repo, revision=revision, allow_patterns=ALLOW))
    except Exception as e:  # noqa: BLE001 - recorded, not hidden
        err.write_text(json.dumps({"repo": repo, "revision": revision, "stage": "download",
                                   "error": f"{type(e).__name__}: {e}"}, indent=2) + "\n")
        print(f"FAILED download {repo}@{revision}: {type(e).__name__}: {e}", file=sys.stderr)
        return False

    # With the Hub unreachable (offline, rate limited), snapshot_download returns
    # whatever is cached locally, which may be a partial snapshot. That is a download
    # problem, never evidence that llama.cpp cannot convert the model.
    if not any((snap / f).is_file() for f in ("config.json", "params.json")):
        present = sorted(p.name for p in snap.iterdir())
        err.write_text(json.dumps({"repo": repo, "revision": revision, "stage": "download",
                                   "error": f"incomplete snapshot (no config.json or params.json): {present}"},
                                  indent=2) + "\n")
        print(f"FAILED download {repo}@{revision}: incomplete snapshot {present}", file=sys.stderr)
        return False

    tmp = gguf.with_suffix(".tmp")
    env = {k: v for k, v in os.environ.items() if k != "NO_LOCAL_GGUF"}  # use the pinned checkout's gguf-py
    env["HF_HUB_OFFLINE"] = "1"
    log = out / f"{stem(repo)}.vocab.log"
    envs = [(Path(sys.executable), "llama.cpp requirements-convert_hf_to_gguf.txt")]
    if tf5_py.is_file():
        envs.append((tf5_py, "fallback: llama.cpp requirements with transformers 5.17.0"))
    attempts: list[dict[str, str]] = []
    converter_env = None
    extra: list[str] = []
    patches: list[str] = []
    model_dir = snap
    with log.open("w") as logf, tempfile.TemporaryDirectory(prefix="gguf-vocab-") as scratch:
        for py, label in envs:
            for _ in range(len(REMEDIES) + 1):
                cmd = [str(py), str(src / "convert_hf_to_gguf.py"), str(model_dir), "--vocab-only", *extra,
                       "--outfile", str(tmp)]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
                logf.write(f"### {label} {patches}: {' '.join(cmd)}\n{proc.stdout}{proc.stderr}\n")
                if proc.returncode == 0 and tmp.exists():
                    converter_env = label
                    break
                tmp.unlink(missing_ok=True)
                attempts.append({"converter_env": label, "args": " ".join(extra), "patches": "; ".join(patches),
                                 "error": (proc.stderr or proc.stdout)[-3000:]})
                remedy = next((r for r in REMEDIES if r[0] in proc.stderr and r[1] not in patches + extra), None)
                if remedy is None:
                    break
                _, what, apply = remedy
                if what.startswith("--"):
                    extra.append(what)
                else:
                    if model_dir == snap:  # patch a private copy, never the HF cache
                        model_dir = Path(scratch) / snap.name
                        shutil.copytree(snap, model_dir, symlinks=False)
                    apply(model_dir)
                    patches.append(what)
            if converter_env is not None:
                break
    if converter_env is None:
        err.write_text(json.dumps({"repo": repo, "revision": revision, "stage": "convert",
                                   "attempts": attempts,
                                   "error": attempts[-1]["error"] if attempts else "no converter env"},
                                  indent=2) + "\n")
        print(f"FAILED convert {repo}@{revision} (see {log})", file=sys.stderr)
        return False
    tmp.replace(gguf)

    template_sha = None
    try:
        sys.path.insert(1, str(src / "gguf-py"))  # the pinned checkout's reader
        import gguf as ggufpy

        reader = ggufpy.GGUFReader(str(gguf))
        field = reader.fields.get("tokenizer.chat_template")
        if field is not None:
            template = bytes(field.parts[field.data[0]]).decode("utf-8")
            template_sha = hashlib.sha256(template.encode()).hexdigest()
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not read chat template from {gguf.name}: {e}", file=sys.stderr)

    meta.write_text(json.dumps({
        "repo": repo,
        "revision": revision,
        "llama_cpp_commit": llama_commit,
        "converter": "convert_hf_to_gguf.py --vocab-only",
        "converter_env": converter_env,
        "converter_args": ["--vocab-only", *extra],
        "converter_patches": patches,
        "sha256": sha256(gguf),
        "chat_template_sha256": template_sha,
    }, indent=2) + "\n")
    print(f"built {gguf.name} ({repo}@{revision})")
    return True


if args == ["--all"]:
    pins = wanted_from_fixtures()
elif not args or "--all" in args:
    sys.exit("usage: gguf_vocab.sh [--force] (--all | ORG/MODEL@REVISION ...)")
else:
    pins = {}
    for a in args:
        repo, sep, rev = a.partition("@")
        if not sep or not rev:
            sys.exit(f"pin a revision: {a}@<commit sha>")
        pins[repo] = rev

def expected_failure(repo: str) -> str | None:
    """Why a recorded failure is expected (and must not fail setup), else None.

    * convert: llama.cpp's converter rejects the model at this pin; the adapter
      reports those fixtures as ``unsupported``.
    * gated: the repo needs an accepted licence plus an HF token. Fixtures pin
      verified ungated tokenizer mirrors (fixtures/llama/family.json), so the
      reference repo's GGUF is not needed for replay.
    Any other download failure (network, rate limit, partial snapshot) is real.
    """
    err = out / f"{stem(repo)}.vocab.error.json"
    if not err.is_file():
        return None
    e = json.loads(err.read_text())
    if e.get("stage") == "convert":
        return "converter does not support this model at the pinned llama.cpp commit"
    if str(e.get("error", "")).startswith("GatedRepoError"):
        return "gated repo (no HF token); fixtures use a verified ungated tokenizer mirror"
    return None


llama_commit = commit()
failed = [repo for repo, rev in sorted(pins.items()) if not build(repo, rev, llama_commit)]
expected = {repo: why for repo in failed if (why := expected_failure(repo))}
for repo, why in expected.items():
    print(f"expected failure: {repo}: {why}", file=sys.stderr)
unexpected = [repo for repo in failed if repo not in expected]
if failed:
    print(f"{len(failed)} of {len(pins)} failed ({len(unexpected)} unexpected): {', '.join(failed)} "
          "(see .engines/gguf/*.vocab.error.json)", file=sys.stderr)
sys.exit(1 if unexpected else 0)
PYEOF
