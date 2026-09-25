"""Offline suite runner: drives an adapter worker and applies the checks.

Flow for ``canitoolcall run --engine E``:

1. load + validate fixtures and families (main env)
2. start ``<engine-python> -m canitoolcall.adapters.worker E`` with
   ``PYTHONPATH=<repo>/src`` (:class:`WorkerClient`), send ``hello``
3. for each fixture send ``replay`` with the strategy ids, receive the
   observations, run :func:`canitoolcall.checks.run_checks`
4. assemble :class:`~canitoolcall.results.RunResults` and write
   ``results/<engine>-<version>.json``

The runner never imports an engine. Harness failures (worker ``ok: false``,
crash, timeout) become ``error`` cases with ``harness_error``; the worker is
restarted after a crash or timeout and the run continues.

``RunConfig.jobs > 1`` runs that many worker processes in parallel. Each
worker is used by one thread at a time, and cases are reported in fixture
order whatever the completion order, so results do not depend on ``jobs``.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import fnmatch
import json
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Literal

from canitoolcall import __version__
from canitoolcall.chunking import DEFAULT_STRATEGIES, ChunkStrategy, TokensPerStep, synthetic_strategies
from canitoolcall.fixtures import Family, Fixture, ValidationIssue, fixtures_digest, load_fixtures, repo_root, validate
from canitoolcall.results import (
    CaseResult,
    EngineInfo,
    Observation,
    ParseResult,
    RunInfo,
    RunResults,
    Status,
)

IncludeObserved = Literal["all", "failures", "none"]


@dataclass(frozen=True)
class RunConfig:
    engine: str
    """Registry name (``vllm``...) or an explicit ``package.module:Class`` adapter spec."""
    fixtures: Sequence[Path] = ()
    """Files/dirs to load; empty -> the default fixtures root."""
    families: Sequence[str] = ()
    """Restrict to these family slugs; empty -> all."""
    strategies: Sequence[ChunkStrategy] = DEFAULT_STRATEGIES
    python: Path | None = None
    """Engine interpreter; None -> :func:`canitoolcall.adapters.engine_python`."""
    out_dir: Path = Path("results")
    include_observed: IncludeObserved = "failures"
    timeout_s: float = 120.0
    """Per-message worker timeout."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for the worker (e.g. HF_HUB_OFFLINE=1). A ``PYTHONPATH``
    entry is appended after the core package path."""
    tags: Sequence[str] = ()
    """Keep fixtures carrying at least one of these tags; empty -> all."""
    ids: Sequence[str] = ()
    """Keep fixtures whose id matches one of these ``fnmatch`` patterns; empty -> all."""
    jobs: int = 1
    """Number of worker processes run in parallel."""
    validate: bool = True
    """Validate the fixtures against the spec first (raises FixtureValidationError)."""
    startup_timeout_s: float = 600.0
    """Timeout for ``hello`` (engine imports can be slow)."""


class WorkerError(RuntimeError):
    """The worker process crashed, timed out, or answered ``ok: false``.

    ``fatal`` is True when the process is gone (crash, timeout, broken pipe)
    and must be restarted before it can serve again. ``unavailable`` is True
    when the worker reported that the engine is not installed.
    """

    def __init__(self, message: str, *, fatal: bool = False, unavailable: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal
        self.unavailable = unavailable


class FixtureValidationError(ValueError):
    """The fixtures do not validate against the spec."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        head = "\n".join(f"  {i}" for i in self.issues[:20])
        more = f"\n  ... and {len(self.issues) - 20} more" if len(self.issues) > 20 else ""
        super().__init__(f"{len(self.issues)} fixture validation issue(s):\n{head}{more}")


# --------------------------------------------------------------------------- worker process

_SHIM_LOCK = threading.Lock()
_SHIM_DIR: Path | None = None


def core_pythonpath() -> Path:
    """Directory to put on the worker's ``PYTHONPATH`` so ``import canitoolcall`` works.

    In a source checkout this is ``<repo>/src``. For an installed package it is
    a private directory holding only a ``canitoolcall`` symlink, so the engine
    interpreter never sees the main environment's other site-packages.
    """
    global _SHIM_DIR
    pkg = Path(__file__).resolve().parent
    root = repo_root()
    if root is not None and pkg.parent == root / "src":
        return pkg.parent
    with _SHIM_LOCK:
        if _SHIM_DIR is None:
            shim = Path(tempfile.mkdtemp(prefix="canitoolcall-pythonpath-"))
            (shim / "canitoolcall").symlink_to(pkg, target_is_directory=True)
            atexit.register(shutil.rmtree, shim, True)
            _SHIM_DIR = shim
        return _SHIM_DIR


def worker_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a worker: the current environment without Python/venv
    overrides, the core ``PYTHONPATH`` (plus ``extra["PYTHONPATH"]``), UTF-8 I/O."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    extra = dict(extra or {})
    paths = [str(core_pythonpath())]
    if extra.get("PYTHONPATH"):
        paths.append(extra.pop("PYTHONPATH"))
    env.update(extra)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class WorkerClient:
    """Talks the JSON-lines protocol of :mod:`canitoolcall.adapters.worker`.

    Usage::

        with WorkerClient("vllm", python=...) as w:
            info = w.hello()
            reply = w.replay(fixture, family, strategies)

    One client drives one process and must not be shared between threads
    without external locking. stderr is drained continuously (its tail is
    attached to crash reports) so a chatty engine can never block the pipe.
    """

    STDERR_TAIL = 60

    def __init__(
        self,
        engine: str,
        python: Path,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 120.0,
        startup_timeout_s: float = 600.0,
    ) -> None:
        self.engine = engine
        self.python = Path(python)
        self.env = dict(env or {})
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: collections.deque[str] = collections.deque(maxlen=self.STDERR_TAIL)
        self._stderr_lock = threading.Lock()
        self._hello: dict[str, Any] | None = None

    # -- lifecycle

    def __enter__(self) -> WorkerClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Launch the worker process (no-op if it is running)."""
        if self.alive:
            return
        if not self.python.exists():
            raise WorkerError(
                f"engine interpreter not found: {self.python} (run scripts/engines/{self.engine}.sh)", unavailable=True
            )
        self._lines = queue.Queue()
        with self._stderr_lock:
            self._stderr.clear()
        self._proc = subprocess.Popen(
            [str(self.python), "-m", "canitoolcall.adapters.worker", self.engine],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=worker_env(self.env),
        )
        assert self._proc.stdout is not None and self._proc.stderr is not None
        threading.Thread(target=self._pump_stdout, args=(self._proc.stdout, self._lines), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(self._proc.stderr,), daemon=True).start()

    @staticmethod
    def _pump_stdout(stream: IO[str], lines: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    def _pump_stderr(self, stream: IO[str]) -> None:
        try:
            for line in stream:
                self._log(line.rstrip("\n"))
        except (OSError, ValueError):
            pass

    def _log(self, line: str) -> None:
        with self._stderr_lock:
            self._stderr.append(line)

    def stderr_tail(self) -> str:
        """The last lines the worker wrote to stderr (plus any non-protocol stdout)."""
        with self._stderr_lock:
            return "\n".join(self._stderr)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()

    def restart(self) -> None:
        """Kill and relaunch the worker process (after a crash or timeout).

        If a ``hello`` succeeded before, it is repeated and the engine version
        must not change.
        """
        self._kill()
        self.start()
        if self._hello is not None:
            before = (self._hello.get("engine"), self._hello.get("version"))
            self._hello = None
            info = self.hello()
            if (info.get("engine"), info.get("version")) != before:
                raise WorkerError(
                    f"engine changed across restart: {before} -> {info.get('engine'), info.get('version')}"
                )

    def close(self) -> None:
        """Send ``shutdown`` and wait; kill on timeout."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                self._send({"op": "shutdown"})
                self._receive(timeout=10.0)
            except WorkerError:
                pass
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        self._kill()

    # -- protocol

    def _send(self, msg: Mapping[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise self._crashed("worker is not running")
        try:
            proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise self._crashed(f"cannot write to worker ({e})") from e

    def _crashed(self, what: str) -> WorkerError:
        proc = self._proc
        code = None
        if proc is not None:
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code = None
        self._kill()
        tail = self.stderr_tail()
        return WorkerError(
            f"{what}; worker exit code {code}" + (f"\n--- worker stderr (tail) ---\n{tail}" if tail else ""),
            fatal=True,
        )

    def _receive(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self._kill()
                raise WorkerError(f"worker timed out after {timeout:g}s", fatal=True)
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise self._crashed("worker exited unexpectedly")
            if not line.strip():
                continue
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                self._log(f"[non-protocol stdout] {line.rstrip()}")
                continue
            if isinstance(reply, dict):
                return reply
            self._log(f"[non-protocol stdout] {line.rstrip()}")

    def request(self, msg: Mapping[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Send one message and return the ``ok`` reply (raises WorkerError otherwise)."""
        self.start()
        self._send(msg)
        reply = self._receive(self.timeout_s if timeout is None else timeout)
        if not reply.get("ok"):
            raise WorkerError(
                str(reply.get("error") or "worker answered ok: false"), unavailable=bool(reply.get("unavailable"))
            )
        return reply

    def hello(self) -> dict[str, Any]:
        """Send ``hello``; returns engine name/version/commit/details."""
        reply = self.request({"op": "hello"}, timeout=self.startup_timeout_s)
        for key in ("engine", "version"):
            if not isinstance(reply.get(key), str) or not reply[key]:
                raise WorkerError(f"malformed hello reply (missing {key!r}): {reply}")
        self._hello = reply
        return reply

    def replay(self, fixture: Fixture, family: Family | None, strategies: Sequence[ChunkStrategy]) -> dict[str, Any]:
        """Send ``replay``; returns the worker reply (raises WorkerError if not ok)."""
        timeout = self.timeout_s if self._hello is not None else max(self.timeout_s, self.startup_timeout_s)
        reply = self.request(
            {
                "op": "replay",
                "fixture": fixture.to_dict(),
                "family": family.to_dict() if family is not None else None,
                "strategies": [s.id for s in strategies],
            },
            timeout=timeout,
        )
        if reply.get("fixture_id") != fixture.id:
            raise WorkerError(f"reply is for fixture {reply.get('fixture_id')!r}, expected {fixture.id!r}")
        return reply


# --------------------------------------------------------------------------- evaluation


def _observation(reply: Mapping[str, Any], synthetic: Collection[str] = ()) -> Observation:
    try:
        streams = reply.get("streams") or {}
        return Observation(
            nonstream=ParseResult.from_dict(reply["nonstream"]),
            streams={str(k): ParseResult.from_dict(v) for k, v in streams.items()},
            synthetic=frozenset(synthetic),
        )
    except (KeyError, TypeError, AttributeError) as e:
        raise WorkerError(f"malformed replay reply ({type(e).__name__}: {e})") from e


def evaluate(
    fixture: Fixture,
    family: Family | None,
    reply: Mapping[str, Any],
    include_observed: IncludeObserved = "failures",
    synthetic: Collection[str] = (),
) -> CaseResult:
    """Turn one worker ``replay`` reply into a CaseResult (runs the checks).

    ``synthetic`` lists the strategy ids that are reported but do not count
    toward the status for this engine (see :func:`canitoolcall.chunking.synthetic_strategies`).
    Raises :class:`WorkerError` if the reply is malformed or the checks fail
    on it (a harness problem, never an engine outcome).
    """
    from canitoolcall.checks import case_status, run_checks

    if not reply.get("supported", False):
        return CaseResult(fixture.id, fixture.family, Status.UNSUPPORTED, reason=reply.get("reason"))
    obs = _observation(reply, synthetic)
    try:
        rows = tuple(run_checks(fixture, family, obs))
    except Exception as e:  # never abort a whole run on one case
        raise WorkerError(f"checks failed on this reply ({type(e).__name__}: {e})") from e
    status = case_status(rows, obs.synthetic)
    keep = include_observed == "all" or (include_observed == "failures" and status is not Status.PASS)
    return CaseResult(
        fixture_id=fixture.id,
        family=fixture.family,
        status=status,
        checks=rows,
        parser_config=reply.get("parser_config"),
        observed=obs if keep else None,
        skipped_strategies=dict(reply.get("skipped") or {}),
    )


def _error_case(fixture: Fixture, message: str) -> CaseResult:
    return CaseResult(fixture.id, fixture.family, Status.ERROR, harness_error=message)


def replay_case(
    client: WorkerClient,
    fixture: Fixture,
    family: Family | None,
    strategies: Sequence[ChunkStrategy],
    include_observed: IncludeObserved = "failures",
    synthetic: Collection[str] = (),
) -> CaseResult:
    """Replay one fixture and evaluate it; harness failures become ``error`` cases.

    A fatal failure (crash, timeout) restarts the worker so the next fixture
    gets a fresh process. If the restart itself fails, the error propagates.
    """
    try:
        return evaluate(fixture, family, client.replay(fixture, family, strategies), include_observed, synthetic)
    except WorkerError as e:
        if e.fatal:
            client.restart()
        return _error_case(fixture, str(e))


# --------------------------------------------------------------------------- run


def select_fixtures(config: RunConfig) -> list[Fixture]:
    """Load the fixtures selected by ``config`` (paths, families, tags, id patterns)."""
    paths = list(config.fixtures) or None
    for p in paths or []:
        if not Path(p).exists():
            raise FileNotFoundError(f"fixtures path does not exist: {p}")
    fixtures = load_fixtures(paths, families=list(config.families) or None, tags=list(config.tags) or None)
    if config.ids:
        fixtures = [f for f in fixtures if any(fnmatch.fnmatchcase(f.id, pat) for pat in config.ids)]
    return fixtures


def family_for(fixture: Fixture, cache: dict[Path, Family | None] | None = None) -> Family | None:
    """The family.json next to the fixture's file (None if missing or unreadable)."""
    if fixture.source is None:
        return None
    path = fixture.source.parent / "family.json"
    if cache is not None and path in cache:
        return cache[path]
    family: Family | None = None
    if path.is_file():
        try:
            family = Family.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError):
            family = None
    if cache is not None:
        cache[path] = family
    return family


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def tokens_per_step(hello: Mapping[str, Any]) -> TokensPerStep:
    """The engine's streaming granularity from a ``hello`` reply (``many`` if not stated)."""
    return "one" if hello.get("tokens_per_step") == "one" else "many"


def _engine_info(hello: Mapping[str, Any]) -> EngineInfo:
    details = dict(hello.get("details") or {})
    if hello.get("pinned_version") is not None:
        details.setdefault("pinned_version", hello["pinned_version"])
    details.setdefault("tokens_per_step", tokens_per_step(hello))
    return EngineInfo(
        name=str(hello["engine"]),
        version=str(hello["version"]),
        commit=hello.get("commit"),
        details=details,
    )


def run(config: RunConfig) -> RunResults:
    """Run the offline suite for one engine and return the results (not written).

    Raises :class:`FixtureValidationError` (invalid fixtures), ``ValueError``
    (nothing selected, bad config) or :class:`WorkerError` (the worker cannot
    start or say hello, or cannot be restarted).
    """
    from canitoolcall.adapters import engine_python
    from canitoolcall.checks import NORMALIZATION

    if config.jobs < 1:
        raise ValueError("jobs must be >= 1")
    if not config.strategies:
        raise ValueError("at least one chunking strategy is required")
    if config.validate:
        for p in config.fixtures:
            if not Path(p).exists():
                raise FileNotFoundError(f"fixtures path does not exist: {p}")
        issues = validate(list(config.fixtures) or None)
        if issues:
            raise FixtureValidationError(issues)
    fixtures = select_fixtures(config)
    if not fixtures:
        raise ValueError("no fixtures selected")
    fam_cache: dict[Path, Family | None] = {}
    families = [family_for(f, fam_cache) for f in fixtures]

    python = config.python or engine_python(config.engine, repo_root())
    started = _now()
    n_workers = min(config.jobs, len(fixtures))
    clients = [
        WorkerClient(config.engine, python, config.env, config.timeout_s, config.startup_timeout_s)
        for _ in range(n_workers)
    ]
    try:
        hellos = [c.hello() for c in clients]
        ident = {(h["engine"], h["version"], h.get("commit")) for h in hellos}
        if len(ident) != 1:
            raise WorkerError(f"workers disagree on the engine: {sorted(map(str, ident))}")
        strategies = list(dict.fromkeys(config.strategies))
        synthetic = synthetic_strategies([s.id for s in strategies], tokens_per_step(hellos[0]))
        if n_workers == 1:
            cases = [
                replay_case(clients[0], fx, fam, strategies, config.include_observed, synthetic)
                for fx, fam in zip(fixtures, families, strict=True)
            ]
        else:
            pool: queue.Queue[WorkerClient] = queue.Queue()
            for c in clients:
                pool.put(c)

            def task(fx: Fixture, fam: Family | None) -> CaseResult:
                client = pool.get()
                try:
                    return replay_case(client, fx, fam, strategies, config.include_observed, synthetic)
                finally:
                    pool.put(client)

            with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="canitoolcall-worker") as ex:
                cases = list(ex.map(task, fixtures, families))
    finally:
        for c in clients:
            c.close()

    hello = hellos[0]
    return RunResults(
        canitoolcall_version=__version__,
        engine=_engine_info(hello),
        run=RunInfo(
            started_at=started,
            finished_at=_now(),
            platform=f"{platform.system().lower()}-{platform.machine()}",
            python=str(hello.get("python") or platform.python_version()),
            fixtures_digest=fixtures_digest(fixtures),
            strategies=tuple(dict.fromkeys(s.id for s in config.strategies)),
            normalization=NORMALIZATION,
            synthetic_strategies=synthetic,
        ),
        cases=tuple(cases),
    )


def run_and_write(config: RunConfig) -> Path:
    """:func:`run`, then write ``<out_dir>/<engine>-<version>.json``; returns the path."""
    results = run(config)
    return results.write(config.out_dir / results.default_filename())


__all__ = [
    "FixtureValidationError",
    "RunConfig",
    "WorkerClient",
    "WorkerError",
    "core_pythonpath",
    "evaluate",
    "family_for",
    "replay_case",
    "run",
    "run_and_write",
    "select_fixtures",
    "tokens_per_step",
    "worker_env",
]
