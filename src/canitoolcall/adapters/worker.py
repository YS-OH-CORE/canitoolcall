"""JSON-lines worker that runs one adapter inside its engine's interpreter.

The runner (main env) starts::

    PYTHONPATH=<repo>/src <engine-python> -m canitoolcall.adapters.worker <engine>

and exchanges one JSON object per line on stdin/stdout. stderr is free-form
logging. Protocol (version 1):

``{"op": "hello"}``
    -> ``{"ok": true, "protocol": 1, "engine": str, "version": str,
    "commit": str|null, "details": {...}}``

``{"op": "replay", "fixture": <fixture record>, "family": <family.json>|null,
"strategies": ["one", "token", "rand:1:8", ...]}``
    -> ``{"ok": true, "fixture_id": str, "supported": bool, "reason": str|null,
    "parser_config": {...}, "nonstream": <ParseResult>, "streams": {id: <ParseResult>}}``
    (``nonstream``/``streams``/``parser_config`` omitted when unsupported)

``{"op": "shutdown"}``
    -> ``{"ok": true}`` then exit 0.

Any harness failure answers ``{"ok": false, "error": "<traceback>"}`` and the
worker keeps serving. Standard library only.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from typing import IO, Any

from canitoolcall.adapters import load_adapter
from canitoolcall.adapters.base import Adapter, ReplayInput
from canitoolcall.chunking import ChunkStrategy, split
from canitoolcall.fixtures import Family, Fixture

PROTOCOL_VERSION = 1


def hello(adapter: Adapter) -> dict[str, Any]:
    return {
        "ok": True,
        "protocol": PROTOCOL_VERSION,
        "engine": adapter.name,
        "version": adapter.version(),
        "commit": adapter.commit(),
        "details": adapter.engine_details(),
    }


def replay(
    adapter: Adapter,
    fixture: Fixture,
    family: Family | None,
    strategies: Sequence[ChunkStrategy],
) -> dict[str, Any]:
    """Replay one fixture: one non-streaming parse plus one stream per strategy.

    Synthetic ``char`` strategies are skipped here: adapters work on token
    ids, and the generic worker never fabricates ids for character splits.
    (An adapter-specific stress path may be added later; see docs/DESIGN.md.)
    """
    raw = ReplayInput.from_fixture(fixture, family)
    sup = adapter.supports(raw.family, raw.model)
    if not sup:
        return {"ok": True, "fixture_id": fixture.id, "supported": False, "reason": sup.reason}
    tools = list(fixture.tools)
    units = adapter.units(raw)
    streams: dict[str, Any] = {}
    for strat in strategies:
        if not strat.realistic:
            continue
        streams[strat.id] = adapter.parse_stream(raw, split(units, strat), tools).to_dict()
    return {
        "ok": True,
        "fixture_id": fixture.id,
        "supported": True,
        "reason": None,
        "parser_config": adapter.parser_config(raw),
        "nonstream": adapter.parse(raw, tools).to_dict(),
        "streams": streams,
    }


def handle(adapter: Adapter, msg: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one protocol message (never raises)."""
    try:
        op = msg.get("op")
        if op == "hello":
            return hello(adapter)
        if op == "replay":
            fam = msg.get("family")
            return replay(
                adapter,
                Fixture.from_dict(msg["fixture"]),
                Family.from_dict(fam) if fam else None,
                [ChunkStrategy.parse(s) for s in msg.get("strategies", [])],
            )
        if op == "shutdown":
            return {"ok": True}
        return {"ok": False, "error": f"unknown op {op!r}"}
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


def serve(adapter: Adapter, stdin: IO[str], stdout: IO[str]) -> int:
    """Serve the protocol until ``shutdown`` or EOF. Returns the exit code."""
    try:
        for line in stdin:
            if not line.strip():
                continue
            msg: Any = None
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                reply: dict[str, Any] = {"ok": False, "error": f"bad JSON: {e}"}
            else:
                reply = (
                    handle(adapter, msg)
                    if isinstance(msg, dict)
                    else {"ok": False, "error": "message must be a JSON object"}
                )
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
            if isinstance(msg, dict) and msg.get("op") == "shutdown":
                break
    finally:
        adapter.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m canitoolcall.adapters.worker <engine>", file=sys.stderr)
        return 2
    # Keep stdout clean for the protocol: engines sometimes print on import.
    proto_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        adapter = load_adapter(args[0])
        return serve(adapter, sys.stdin, proto_out)
    finally:
        sys.stdout = proto_out


if __name__ == "__main__":
    raise SystemExit(main())
