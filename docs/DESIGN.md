# CanIToolCall design

This document is the contract between the people (and agents) building CanIToolCall in parallel. It covers the architecture, the interfaces, the faithfulness rules found by the engine spike, and **exactly which files each builder owns**. The plan is in `../docs/plans/01-canitoolcall.md` (outside this repo); the fixture format is in [`spec/README.md`](../spec/README.md); per-family format notes are in [`docs/formats/`](formats/README.md).

## 1. What the tool does

CanIToolCall replays recorded raw model outputs (**fixtures**) through each inference engine's **own parser code**, offline, with no GPU and no model weights. Each fixture is parsed once without streaming and once per chunking strategy. The results are checked against the expected parse and against each other. Results are written as JSON and rendered into a static family × engine matrix.

An adapter counts only if it runs the engine's genuine parser code at a pinned version. If something cannot run, it is reported as `unsupported`, never guessed.

## 2. Architecture

```
fixtures/<family>/*.jsonl ──► fixtures.py (load + validate)
                                   │
canitoolcall run --engine E ──► runner.py ── WorkerClient ──(JSON lines over stdin/stdout)──┐
   (main dev env, .venv)           │                                                        │
                                   │                   .venvs/E/bin/python -m canitoolcall.adapters.worker E
                                   │                          │   (PYTHONPATH=<repo>/src, stdlib-only core)
                                   │                          ▼
                                   │                   adapters/E.py (Adapter) ──► engine parser code
                                   │                          │     (vLLM/SGLang/transformers: in-process Python;
                                   │                          │      llama.cpp/Ollama: compiled harness subprocess)
                                   ▼                          │
                          checks.py ◄── Observation (nonstream + one ParseResult per strategy)
                                   │
                          results.py ── results/E-<version>.json ──► matrix.py ──► site/_build/index.html
```

### Process model and import rules

- **The main env** (`uv sync`, Python 3.12, `.venv/`) runs the CLI, the runner, the checks, validation, probe and matrix. It never imports an engine.
- **Each engine runs in its own isolated interpreter**, `.venvs/<engine>/bin/python`, which is built by `scripts/engines/<engine>.sh`. Override it with `$CANITOOLCALL_<ENGINE>_PYTHON`. The runner starts `python -m canitoolcall.adapters.worker <engine>` there with `PYTHONPATH=<repo>/src`, so canitoolcall is **not installed** into engine venvs.
- Therefore these modules **must import with the standard library only**: `canitoolcall/__init__.py`, `fixtures.py` (jsonschema is imported lazily inside `validate`), `chunking.py`, `results.py`, `adapters/__init__.py`, `adapters/base.py`, `adapters/worker.py` and every `adapters/<engine>.py` at module level. Engine imports go inside methods. `tests/core/test_adapters_registry.py::test_worker_starts_in_engine_env` enforces this.
- **Worker protocol v1** (see `adapters/worker.py` for the full text): `hello`, `replay` (fixture record + family + strategy ids → `nonstream` + `streams{id: ParseResult}` + `parser_config`), and `shutdown`. One worker process serves a whole run. A harness failure answers `{"ok": false, "error": traceback}`, and the worker keeps serving.
- The compiled harnesses (llama.cpp C++, Ollama Go) are separate long-lived processes, owned by their Python adapter, and speak **JSON lines** as well: one process per run, never one per fixture.

### Why fixtures are token ids first

Engines stream **tokens**, and special tokens are atomic. The spike showed that per-character splits inside a special token give failures no real server can produce, and that re-encoding text is lossy (`mistral_common` never maps the text `[TOOL_CALLS]` to its control token). So:

1. Fixtures carry `output_token_ids` plus a `tokenizer` pin whenever the source allows it (spec/README.md, "Token ids come first").
2. `Adapter.units(raw)` returns those ids (or the engine tokenizer's encoding of `raw_output` as a fallback).
3. `chunking.split(units, strategy)` groups the ids. The default strategies are `one`, `token` and `rand:1..5:8`, all seeded and identical on every machine.
4. The adapter turns each id group into a text delta with **the engine's own detokenizer**, then calls the engine's streaming parser.

`char:<seed>` exists as an opt-in stress strategy. The generic worker skips it, because it cannot be expressed as token groups. It is reported separately and never counts toward matrix status.

## 3. Interfaces (core-owned)

The signatures below are frozen for the parallel build. If you need a change, don't edit core files: describe the change in your final report, and the core owner applies it.

| Module | Key types / functions | State in skeleton |
|---|---|---|
| `fixtures.py` | `Fixture`, `Family`, `ReferenceModel`, `Provenance`, `Expected`, `ExpectedError`, `load_fixtures`, `load_families`, `validate`, `fixtures_digest`, `default_fixtures_dir` | **implemented** + tested |
| `chunking.py` | `ChunkStrategy` (`id`, `parse`, `realistic`), `DEFAULT_STRATEGIES`, `group_sizes`, `split`, `parse_strategies` | **implemented** + tested (golden grouping pinned) |
| `results.py` | `Status`, `worst_status`, `ParsedToolCall`, `ParseResult`, `Observation`, `CheckResult`, `CaseResult`, `EngineInfo`, `RunInfo`, `RunResults` (`to_dict`/`from_dict`/`write`/`load`) | **implemented** + tested against `spec/results.schema.json` |
| `adapters/base.py` | `Adapter` ABC (`name`, `pinned_version`, `version`, `supports`, `parser_config`, `units`, `parse`, `parse_stream`, `engine_details`, `commit`, `close`), `ReplayInput`, `Support`, `AdapterUnavailable` | **implemented** |
| `adapters/__init__.py` | `ENGINES` registry, `adapter_class`, `load_adapter`, `engine_python` | **implemented** |
| `adapters/worker.py` | protocol v1: `hello`, `replay`, `handle`, `serve`, `main` | **implemented** + tested with a fake adapter |
| `checks.py` | `normalize_text`, `canonical`, `compare`, `check_*` (8 checks), `ALL_CHECKS`, `run_checks` | stubs |
| `runner.py` | `RunConfig`, `WorkerClient`, `WorkerError`, `evaluate`, `run`, `run_and_write` | stubs |
| `cli.py` | `run`, `probe`, `matrix`, `validate`, `engines` | **implemented** (dispatches to stubs; a stub exits with code 2 and a "not implemented yet" message) |
| `probe.py` | `Scenario`, `ProbeExpectation`, `ProbeOutcome`, `ProbeReport`, `BUILTIN_SCENARIOS`, `chat`, `evaluate`, `probe` | stubs |
| `pytest_plugin.py` | options, `pytest_generate_tests`, `assert_conforms` | options done; rest stubs |
| `matrix.py` | `Cell`, `Matrix`, `load_results`, `build_matrix`, `render_site` | stubs |

### Adapter contract (every adapter MUST honour these)

Each rule was found by the spike **breaking without it**:

1. **Token ids first**: see §2.
2. **Engine detokenization:**
   - **vLLM:** call `parser.adjust_request(request)` first, since many parsers force `skip_special_tokens=False` there. Then detokenize with the v1 `IncrementalDetokenizer` (`from_new_request(tok, SimpleNamespace(request_id, sampling_params=req.to_sampling_params(4096, {}), prompt_token_ids, prompt_embeds=None))`, then `.update(ids, False)` and `.get_next_output_text(finished, delta=True)`). Never use `tok.decode`: vLLM's `MistralTokenizer` keeps `[TOOL_CALLS]`/`[ARGS]` even when `skip_special_tokens=True`.
   - **SGLang:** when tools are present, detokenize with `skip_special_tokens=False`.
   - **llama.cpp and Ollama:** render only the harness's `preserved_tokens` as text, from a **vocab-only GGUF** made by `convert_hf_to_gguf.py --vocab-only`, using `common_token_to_piece(vocab, id, special = id ∈ preserved_tokens)`. **Never** approximate GGUF CONTROL tokens with HF `special` flags: that is wrong for Gemma 4's `<|"|>`.
3. **Serving-layer entry points:**
   - **vLLM:** `ParserManager.get_parser(...)` gives the parser class. Non-streaming: `p.parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)`. Streaming: `p.parse_delta(delta_text, delta_ids, request, prompt_token_ids=prompt_ids, finished=…)`. **`prompt_token_ids` is required**; without it `reasoning_ended` is never set and calls come out as content.
   - **SGLang:** run `ReasoningParser` first, then `FunctionCallParser` on the normal text, and call `parse_stream_end` at finish.
   - **llama.cpp:** re-parse the accumulated text with `is_partial=true` after each chunk, then `compute_diffs`, exactly like `llama-server`.
   - **Ollama:** `ParserForName`, `Init` once, then `Add(chunk, done)`, as `server/routes.go` does.
4. **Prompt context.** Build `prompt_token_ids` from the reference model's chat template with the fixture's tools and `add_generation_prompt=True`. `ReplayInput.generation_prompt` covers pre-filled reasoning openers such as `<think>\n`.
5. **Stop tokens.** `raw_output` ends **before** the stop token (`ReplayInput.stop_tokens`, from `generation_config.json`). If an engine needs the stop token in the text (SGLang's gpt-oss detector needs `<|call|>`), the adapter re-appends exactly what that engine's server would keep, and records it in `parser_config`.
6. **Engine exceptions are outcomes.** Catch them and return `ParseResult(exception="Type: msg")`. Raise only for harness problems; the runner records those as `error`.
7. **Pin everything.** `parser_config(raw)` records the tool/reasoning parser names, template source + sha256, `enable_thinking`/`reasoning_format`, tokenizer mode and repo@revision. For llama.cpp, the template **source** matters: the current HF DeepSeek-V3.1 template fails where llama.cpp's own `models/templates/` copy works.
8. **Stream accumulation** is OpenAI-client style. Concatenate content and reasoning deltas, and merge tool-call deltas by `index`: the name is set once, and argument fragments are concatenated into `arguments_raw`.
9. **`supports(family, model)` is honest and cheap.** If the engine has no parser for a model version (for example, SGLang 0.5.20 has none for DeepSeek-V4.1), return `Support(False, reason)`. The runner reports `unsupported`.
10. **Offline by default.** Tokenizers and templates are fetched once into the HF cache at pinned revisions (config and tokenizer files only, never weights). Tests set `HF_HUB_OFFLINE=1` when the cache is warm.

### Checks and normalization

The semantics are in spec/README.md ("Checks" and "Normalization policy soft-v1"). A case's status is the worst over its checks: `fail` > `error` > `soft_pass` > `pass`. `unsupported` is used when the adapter declines. Whitespace-only differences in content/reasoning, and leading or trailing newlines, are `soft_pass`, never `pass`, and the matrix shows them separately. The spike saw these in several engines (glm45 leading `\n`, Qwen3 hermes `"\n\n"` content, Gemma 4 trailing `\n`).

## 4. File ownership for parallel builders

A builder edits **only** the files listed for it. The core owner edits shared config. Everyone may read everything, including `.spikes/` (the gitignored spike code, which is the reference implementation for every adapter).

| Builder | Owns (create/edit) | Must not touch |
|---|---|---|
| **core** | `src/canitoolcall/{__init__,__main__,fixtures,chunking,checks,results,runner,cli}.py`, `src/canitoolcall/adapters/{__init__,base,worker}.py`, `tests/core/**`, `tests/conftest.py`, `spec/**`, `pyproject.toml`, `uv.lock`, `.gitignore`, `.python-version`, `docs/DESIGN.md` | adapters' engine modules, fixtures |
| **fixtures group N** | `fixtures/<slug>/**` for its families (`family.json` + `*.jsonl`), `scripts/fixtures/<slug>/**` (generator/import scripts), `docs/formats/<slug>.md` for its families | other groups' slugs, `spec/`, `src/` |
| **adapter `<engine>`** | `src/canitoolcall/adapters/<engine>.py`, `tests/adapters/test_<engine>.py`, `scripts/engines/<engine>.sh`, `harnesses/<engine>/**` (compiled engines only) | other adapters, core modules, `ENGINES` registry |
| **probe + plugin** | `src/canitoolcall/{probe,pytest_plugin}.py`, `tests/probe/**` | checks.py (call it, don't edit it) |
| **site + ci + docs** | `src/canitoolcall/matrix.py`, `site/templates/**`, `tests/site/**`, `.github/workflows/**`, `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `docs/PUBLISHING.md` | results format (read it via `results.py`) |

Shared-resource rules:
- **Vocab-only GGUFs** are generated by `scripts/engines/gguf_vocab.sh` into `.engines/gguf/<org>--<model>.vocab.gguf`. The llama.cpp builder owns that script and the Ollama builder consumes it read-only. It needs a llama.cpp checkout for `convert_hf_to_gguf.py`.
- **Engine checkouts, unpacked wheels and harness builds** go under `.engines/<engine>/` (gitignored), never `.spikes/`.
- **The fixture shape** is defined by `spec/`. A fixture builder who needs a new field or tag asks core. Project-specific tags use `x-*` and need no approval.
- **Family slugs** (directory names) are fixed: `qwen3-hermes`, `qwen3-xml`, `gpt-oss`, `deepseek`, `kimi`, `glm`, `llama`, `mistral`, `gemma4`. Adapters key their parser maps on these.
- Before handing off, every builder runs `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy && uv run pytest`.

## 5. Builder briefs

### core
Implement `checks.py` (all 8 checks, soft-v1, leakage against `family.markers`, lazy `jsonschema`) and `runner.py` (`WorkerClient` with timeouts and restart-after-crash, `evaluate`, `run`, `run_and_write`; the run metadata is timestamps, platform, fixtures digest, strategy ids and normalization). Unit-test checks with hand-built `ParseResult`s. Test the runner end to end with a fake-engine worker (for example via `$CANITOOLCALL_<ENGINE>_PYTHON` pointing at the dev interpreter and a test adapter).

### fixture groups
Aim for ~22 fixtures per family (≥150 overall, ~200 target). Every fixture needs mandatory provenance, and every family needs a `family.json` (markers, reference models at pinned revisions, stop tokens, generation prompt). Sources, in order of preference:
1. **Generator scripts in `scripts/fixtures/<slug>/`** that render through the official template or encoder with `apply_chat_template(tokenize=True)` and store `output_token_ids`. They run in `.venvs/transformers`, or in a venv with `openai-harmony`/`mistral_common` for those families.
2. **Engine test suites**, with the license and a line-anchored URL: vLLM/SGLang/transformers (Apache-2.0), llama.cpp `tests/test-chat.cpp` and Ollama `model/parsers/*_test.go` (MIT).
3. **Bug reports**, with the issue URL.

Cover the edge-case tags in spec/README.md. **History render ≠ generation**: see docs/formats and spec/README.md. Mistral `[CALL_ID]` and the gpt-oss role-header recipient must come from engine tests or real generations, not from history renders. **Never type a format by hand.**

| Group | Families | Why grouped |
|---|---|---|
| fixtures-1 | `deepseek`, `qwen3-hermes`, `gemma4` | one heavy family (5 DeepSeek formats, DSML encoders) + two easy ones |
| fixtures-2 | `gpt-oss`, `mistral`, `qwen3-xml` | two encoder-library families (openai-harmony, mistral_common) + one moderate one |
| fixtures-3 | `kimi`, `glm`, `llama` | K2 + K3 encoder, GLM variants, and Llama (gated: use documented template mirrors or engine tests) |

### adapters
The reference implementations are in `.spikes/<engine>/`. The spike's results files list the observed candidate discrepancies. Don't report those upstream until they are triaged with fixtures.

| Engine | Pin | Env | Notes |
|---|---|---|---|
| vllm | 0.30.0 | `.venvs/vllm` (py3.12; manylinux aarch64 wheel unzipped under `.engines/vllm/`, added via `.pth`; CPU torch) | `ParserManager` path; the Harmony path uses `is_harmony`; `tokenizer_mode='mistral'` for Mistral |
| sglang | 0.5.20 | `.venvs/sglang` (same unzip + `.pth` approach) | text-only parsers; the gpt-oss detector needs `<|call|>` |
| llamacpp | `a25c9865` | `.engines/llamacpp/` clone + harness build (cmake, ~16 s); `.venvs/llamacpp` for `convert_hf_to_gguf.py` | batch JSON-lines mode; exact vocab-GGUF detokenization; pin the template source |
| transformers | 5.17.0 | `.venvs/transformers` | supported only where the Hub repo ships `response_template` (Gemma 4 today) |
| ollama (stretch) | `7af39318` | `.engines/ollama/` clone + `go build` (GOTOOLCHAIN=auto) | built-in parsers first; the legacy `tools.NewParser` path is a stretch goal; reuse the vocab GGUFs |

Each adapter test file must contain:
- a fast unit test of `supports`/`parser_config` that doesn't need the engine;
- `@pytest.mark.engine("<engine>")` tests that replay the core sample and at least one fixture per supported family through the real worker.

### probe + plugin
`probe.py`: stdlib `urllib`, hand-parsed SSE, 6–8 structural scenarios, stream and non-stream, and never log the API key. Test against a local stub HTTP server in `tests/probe/`. `pytest_plugin.py`: parametrize over the bundled corpus by id and implement `assert_conforms` via `checks.compare`. The plugin must stay inert for unrelated test runs (it is installed via `pytest11`).

### site + ci + docs
- `matrix.py`: Jinja2 templates in `site/templates/`, built **only** from `results/*.json`. Its tests use results files generated by `RunResults` in the test.
- Workflows:
  - `ci.yml`: lint, types, tests on PRs, plus `uv build` and `twine check`.
  - `nightly.yml`: engine setup scripts, `canitoolcall run` per engine, `canitoolcall matrix`, and a Pages artifact.
- Nothing is pushed or published from this machine. `docs/PUBLISHING.md` lists the human steps.

## 6. Candidate discrepancies from the spike (untriaged)

These must be turned into fixtures and re-verified before anyone reports them upstream:
- vLLM `deepseek_v31`, SGLang `hermes`/`deepseekv31`: the call is lost or has empty arguments when the whole output arrives in one delta.
- SGLang `glm45`: `\n<think>` leaks into `reasoning_content`.
- llama.cpp: the current HF DeepSeek-V3.1 template yields malformed `preserved_tokens`.
- Ollama `qwen3-thinking`: whether `thinking` has a leading newline depends on the chunking.
- Mistral: history renders contain `[CALL_ID]`, which parsers don't expect. This is probably not a bug, since the model doesn't generate it.

## 7. Definition of done (v0.1)

- At least 2 real engine adapters run their genuine parser code against ≥150 fixtures. Target: vLLM, SGLang and llama.cpp.
- The matrix site is generated only from real `results/*.json`.
- `uv build` and `twine check` pass.
- README, CONTRIBUTING, the spec, LICENSE and the CI workflows are present.
- Nothing is published.
