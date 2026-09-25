# Results snapshot, 2026-09-25

Real offline replays of the whole fixture corpus (469 fixtures, 9 families, `fixtures_digest` `a1e7b553…`) through five engines' own parser code. The runs were on macOS arm64 with the engine workers on Python 3.12.13. Nothing here is hand-written except the triage selectors in `triage.py`.

| File | Engine (pin) | pass | soft_pass | fail | error | unsupported |
|---|---|---|---|---|---|---|
| `vllm-0.30.0.json.gz` | vLLM 0.30.0 | 268 | 86 | 115 | 0 | 0 |
| `sglang-0.5.20.json.gz` | SGLang 0.5.20 | 189 | 62 | 197 | 0 | 21 |
| `llamacpp-a25c9865.json.gz` | llama.cpp `a25c9865` | 278 | 46 | 114 | 0 | 31 |
| `ollama-7af39318.json.gz` | Ollama `7af39318` | 224 | 10 | 27 | 0 | 208 |
| `transformers-5.17.0.json.gz` | transformers 5.17.0 | 33 | 7 | 8 | 0 | 421 |

Which chunking strategies count depends on the engine. llama-server, Ollama and transformers `serve` stream one token per event, so for those three only `token` (plus the non-streaming parse) counts; `one`, `special` and `rand:*:8` are run and shown but recorded in `run.synthetic_strategies`. `char:*` never counts.

These files replace the first runs of the same day, after two harness errors were fixed: fixtures tagged `truncated` (cut by `max_tokens`) had a stop token appended, and multi-token chunkings counted for one-token-per-step engines. Together they caused six false failures (SGLang `gpt-oss/harmony-truncated-in-arguments`; llama.cpp `kimi/k3-truncated-in-reasoning` and `kimi/k3-bug-truncated-reasoning-recorded`; three Ollama cases).

Each engine was run with:

```bash
bash scripts/engines/<engine>.sh
uv run canitoolcall run --engine <engine> -j 3 --env HF_HUB_OFFLINE=1 --out results/2026-09-25 \
  --strategy one --strategy special --strategy token \
  --strategy rand:1:8 --strategy rand:2:8 --strategy rand:3:8 --strategy rand:4:8 --strategy rand:5:8 \
  --strategy char:0 --strategy char:1
```

The files are gzipped to keep the repository small. `canitoolcall matrix --results results/2026-09-25` reads them directly.

## Triage

`triage.jsonl` assigns every failing case, per engine, to exactly one finding. `tests/core/test_results_snapshot.py` checks this against the results files. `triage.py` regenerates it. The classifications are:

| Classification | Meaning |
|---|---|
| `engine_bug` | The engine's parser returns a wrong parse for output the model can produce. One finding can cover many fixtures: count findings, not fixtures (for example `sglang-gpt-oss-role-header-recipient` is one gap across every `x-recipient-in-role` fixture). |
| `engine_bug_known_upstream` | The same, and already reported upstream (see `upstream`). |
| `engine_bug_candidate` | Reproduced through the engine's own code path, but not yet confirmed against a live server. |
| `truncation_policy` | Output cut by `max_tokens` comes back as a (partial) tool call, which the spec treats as a failure (spec/README.md, "Expected results"). |
| `strict_grammar_variant` | The engine rejects an engine-test variant that the official template never renders. |
| `not_generated_by_model` | The input is not what the model generates, for example `<tool_call>` without the newline Qwen3 emits. |
| `format_unverified` | Whether the model generates this form is unverified, for example Mistral v11 `[CALL_ID]` (mistral-common's generation grammar omits it, llama.cpp's tests assume it). Not counted as a bug until a recorded generation settles it. |
| `not_applicable_stop_applied` | The fixture contains an interior stop token that a server applying the stop token never passes on. |
| `intended_engine_behaviour` | The engine's own test asserts this behaviour; the fixture disagrees. |
| `uncertain_history_render` / `contested_fixture` | The expected parse depends on a history render or on a disputed format. Not counted as a bug. |
| `untriaged_discrepancy` | A failing case that is not yet assigned to a root cause. |

Every finding has a `repro` command that replays one of its fixtures through the harness. Four findings also have `direct_repro` scripts in `repro/`, which call the engine's own API with no canitoolcall code. Run each one with that engine's interpreter, for example `HF_HUB_OFFLINE=1 .venvs/sglang/bin/python results/2026-09-25/repro/sglang_qwen25_one_delta.py`.

This project has reported nothing upstream. Findings classified `engine_bug_known_upstream` match an issue someone else already filed, linked in `upstream`. Draft reports for the other findings are in `RELEASE_CHECKLIST.md` §7. Re-check each finding against the engine's latest release before filing it.
