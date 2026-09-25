"""Command-line interface: ``canitoolcall {run,probe,matrix,validate,engines}``.

* ``run --engine E``        offline suite: replay fixtures through engine E's parsers
* ``probe --base-url --model``  live check of an OpenAI-compatible endpoint
* ``matrix``               build the static matrix site from ``results/*.json``
* ``validate [PATH...]``   validate fixtures and family.json against the spec
* ``engines``              list known engines and the interpreter each would use

Exit codes: 0 success; 1 conformance failures (run/probe) or invalid fixtures
(validate); 2 usage or harness error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from canitoolcall import __version__

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree (exposed for tests and docs)."""
    from canitoolcall.adapters import ENGINES

    p = argparse.ArgumentParser(
        prog="canitoolcall",
        description="Conformance suite for tool-call and reasoning parsers across inference engines.",
    )
    p.add_argument("--version", action="version", version=f"canitoolcall {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    r = sub.add_parser("run", help="replay fixtures through an engine's parsers (offline)")
    r.add_argument(
        "--engine",
        required=True,
        type=_engine_arg,
        metavar="ENGINE",
        help=f"one of {', '.join(sorted(ENGINES))}, or an adapter spec package.module:Class",
    )
    r.add_argument(
        "--fixtures",
        nargs="*",
        type=Path,
        default=[],
        metavar="PATH",
        help="fixture files/dirs (default: the fixtures root)",
    )
    r.add_argument("--family", action="append", default=[], metavar="SLUG", help="only this family (repeatable)")
    r.add_argument(
        "--strategy",
        action="append",
        default=[],
        metavar="ID",
        help="chunking strategy id, e.g. one, token, rand:1:8 (repeatable; default set if omitted)",
    )
    r.add_argument(
        "--python",
        type=Path,
        default=None,
        help="engine interpreter (default: $CANITOOLCALL_<ENGINE>_PYTHON or .venvs/<engine>)",
    )
    r.add_argument("--out", type=Path, default=Path("results"), help="results directory (default: results/)")
    r.add_argument(
        "--observed",
        choices=["all", "failures", "none"],
        default="failures",
        help="which cases keep the observed parses in the results file",
    )
    r.add_argument("--timeout", type=float, default=120.0, help="per-fixture worker timeout in seconds")
    r.add_argument("--tag", action="append", default=[], metavar="TAG", help="only fixtures with this tag (repeatable)")
    r.add_argument(
        "--id", action="append", default=[], metavar="PATTERN", help="only fixture ids matching this glob (repeatable)"
    )
    r.add_argument("--jobs", "-j", type=_positive_int, default=1, help="worker processes to run in parallel")
    r.add_argument(
        "--env",
        action="append",
        default=[],
        type=_env_arg,
        metavar="KEY=VALUE",
        help="extra environment for the worker, e.g. HF_HUB_OFFLINE=1 (repeatable)",
    )
    r.add_argument("--no-validate", action="store_true", help="skip validating the fixtures against the spec")

    pr = sub.add_parser("probe", help="check a live OpenAI-compatible endpoint")
    pr.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    pr.add_argument("--model", required=True)
    pr.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable holding the API key (never pass keys on the command line)",
    )
    pr.add_argument("--no-stream", action="store_true", help="skip the streaming variants")
    pr.add_argument("--timeout", type=float, default=60.0)
    pr.add_argument("--concurrency", type=_positive_int, default=4, help="parallel requests (default 4)")
    pr.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the report as JSON ('-' writes JSON to stdout and the table to stderr)",
    )

    m = sub.add_parser("matrix", help="build the static matrix site from results files")
    m.add_argument("--results", type=Path, default=Path("results"))
    m.add_argument("--out", type=Path, default=Path("site/_build"))
    m.add_argument("--templates", type=Path, default=None)
    m.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="fixture corpus for drill-down pages (default: $CANITOOLCALL_FIXTURES or the repo's fixtures/)",
    )

    v = sub.add_parser("validate", help="validate fixtures against the spec")
    v.add_argument("paths", nargs="*", type=Path, help="fixture files/dirs (default: the fixtures root)")

    sub.add_parser("engines", help="list known engines and their interpreters")
    return p


def _engine_arg(value: str) -> str:
    from canitoolcall.adapters import ENGINES, is_adapter_spec

    if value in ENGINES or is_adapter_spec(value):
        return value
    raise argparse.ArgumentTypeError(f"unknown engine {value!r} (choose from {', '.join(sorted(ENGINES))})")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _env_arg(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    return key, val


def cmd_run(args: argparse.Namespace) -> int:
    from canitoolcall.chunking import parse_strategies
    from canitoolcall.runner import FixtureValidationError, RunConfig, WorkerError, run

    try:
        cfg = RunConfig(
            engine=args.engine,
            fixtures=tuple(args.fixtures),
            families=tuple(args.family),
            strategies=parse_strategies(args.strategy),
            python=args.python,
            out_dir=args.out,
            include_observed=args.observed,
            timeout_s=args.timeout,
            env=dict(args.env),
            tags=tuple(args.tag),
            ids=tuple(args.id),
            jobs=args.jobs,
            validate=not args.no_validate,
        )
        results = run(cfg)
    except (FixtureValidationError, WorkerError, ValueError, FileNotFoundError) as e:
        print(f"canitoolcall run: {e}", file=sys.stderr)
        return EXIT_ERROR
    path = results.write(cfg.out_dir / results.default_filename())
    summary = results.summary()
    totals = summary["totals"]
    print(f"{results.engine.name} {results.engine.version}: {len(results.cases)} case(s) -> {path}")
    for fam, counts in summary["by_family"].items():
        print(f"  {fam:14} " + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
    print("  total          " + "  ".join(f"{k}={v}" for k, v in totals.items()))
    return EXIT_FAILURES if totals["fail"] or totals["error"] else EXIT_OK


def cmd_probe(args: argparse.Namespace) -> int:
    import os

    from canitoolcall.probe import probe

    report = probe(
        args.base_url,
        args.model,
        api_key=os.environ.get(args.api_key_env),
        stream_modes=(False,) if args.no_stream else (False, True),
        timeout_s=args.timeout,
        concurrency=args.concurrency,
    )
    to_stdout = args.json is not None and str(args.json) == "-"
    print(report.render_text(), file=sys.stderr if to_stdout else sys.stdout)
    if args.json is not None:
        blob = json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"
        if to_stdout:
            sys.stdout.write(blob)
        else:
            args.json.write_text(blob, encoding="utf-8")
    return EXIT_FAILURES if any(o.status in ("fail", "error") for o in report.outcomes) else EXIT_OK


def cmd_matrix(args: argparse.Namespace) -> int:
    from canitoolcall.matrix import render_site

    try:
        index = render_site(args.results, args.out, args.templates, fixtures_dir=args.fixtures)
    except (FileNotFoundError, ValueError) as e:
        print(f"canitoolcall matrix: {e}", file=sys.stderr)
        return EXIT_ERROR
    print(f"wrote {index}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    from canitoolcall.fixtures import iter_fixture_files, validate

    paths = args.paths or None
    missing = [p for p in paths or [] if not p.exists()]
    if missing:
        print(f"canitoolcall validate: no such path: {', '.join(map(str, missing))}", file=sys.stderr)
        return EXIT_ERROR
    files = list(iter_fixture_files(paths))
    issues = validate(paths)
    for issue in issues:
        print(issue, file=sys.stderr)
    print(f"{len(files)} file(s) checked, {len(issues)} issue(s)")
    return EXIT_FAILURES if issues else EXIT_OK


def cmd_engines(args: argparse.Namespace) -> int:
    from canitoolcall.adapters import ENGINES, adapter_class, engine_python
    from canitoolcall.fixtures import repo_root

    for name in sorted(ENGINES):
        cls = adapter_class(name)
        print(f"{name:13} pinned={cls.pinned_version:42} python={engine_python(name, repo_root())}")
    return EXIT_OK


COMMANDS = {
    "run": cmd_run,
    "probe": cmd_probe,
    "matrix": cmd_matrix,
    "validate": cmd_validate,
    "engines": cmd_engines,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except NotImplementedError as e:
        print(f"canitoolcall {args.command}: not implemented yet ({e or 'stub'})", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
