# Contributing to CanIToolCall

Thank you for helping. Most contributions fall into one of three kinds, and each fits in a single PR:

1. **Add a model family.** This is the most common kind, and the rest of this guide centres on it.
2. **Add fixtures to an existing family**, for example a new edge case or a regression from a bug report.
3. **Add or update an engine adapter.**

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately, as described in [SECURITY.md](SECURITY.md), not in public issues.

## Development setup

```sh
uv sync                                   # main env: CLI, runner, checks, matrix (Python 3.12)
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts && uv run mypy && uv run pytest
```

CI runs exactly that command, plus `canitoolcall validate`, `uv build` and `twine check`. Engine tests (`@pytest.mark.engine("vllm")`) skip themselves until you build that engine's environment with `bash scripts/engines/<engine>.sh`. That script puts the engine in an **isolated** venv under `.venvs/<engine>/`. Never install vLLM, SGLang or torch into the main env.

Read [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture, and [`spec/README.md`](spec/README.md) for the fixture format.

## Add a model family: one PR

Suppose the new family's slug is `acme`, for a model `acme-ai/Acme-3`. The PR contains these files:

| File | What it holds |
|---|---|
| `docs/formats/acme.md` | the format notes: the exact raw text, the special tokens, the source template URL and revision, the engine parsers that handle the format (with both vLLM's and SGLang's names), and known bugs with issue URLs |
| `fixtures/acme/family.json` | the family metadata |
| `fixtures/acme/*.jsonl` | the fixtures, grouped by topic (`basic.jsonl`, `parallel.jsonl`, `edge.jsonl`, `truncated.jsonl`, …) |
| `scripts/fixtures/acme/build.py` | the generator that reproduces every `template_render` fixture |
| `src/canitoolcall/adapters/<engine>.py` | only if an adapter needs a parser mapping for `acme` |

### Step by step

1. **Write the format notes first.** Render an assistant tool-call message through the model's **official** chat template or encoder, for example with `apply_chat_template` or the vendor's encoder library. Paste the output verbatim into `docs/formats/acme.md` and record the template URL at a pinned revision. Look up which parser each engine uses for it. [`docs/formats/qwen3-hermes.md`](docs/formats/qwen3-hermes.md) is a good model.

2. **Write `fixtures/acme/family.json`** (schema: [`spec/family.schema.json`](spec/family.schema.json)):
   - `slug`, `name`, `spec_version` and `has_reasoning`
   - `markers`: every special string that must never leak into content, reasoning or arguments. The `no_leakage` check uses this list.
   - `reference_models`: the models whose tokenizer and template the adapters use, each with:
     - `repo` and `revision` (a full commit sha)
     - `stop_tokens`, from `generation_config.json`
     - `default_generation_prompt`, when `add_generation_prompt` pre-fills text such as `<think>\n`
   - `format_notes`: `docs/formats/acme.md`

3. **Write the generator, `scripts/fixtures/acme/build.py`.** Follow the "Token ids come first" rule in the spec:
   1. Render with `apply_chat_template(tokenize=True)`.
   2. Slice off the prompt ids.
   3. Cut at the first stop id.
   4. Store the result as `output_token_ids`, with a `tokenizer` pin.
   5. Set `raw_output` to those ids decoded with `skip_special_tokens=False`.

   Download only config and tokenizer files at pinned revisions, never weights. The script must be deterministic: re-running it must leave `git diff` clean. [`scripts/fixtures/qwen3-hermes/build.py`](scripts/fixtures/qwen3-hermes/build.py) is a complete example.

4. **Cover the edge cases.** Aim for about 20 fixtures and use the tags from the spec:
   - `single-call`, `parallel-calls` and `no-call`
   - `nested-json`, `unicode`, `string-escapes` and `numeric-arguments`
   - `empty-arguments` and `marker-in-arguments`
   - `text-before-call`, `text-after-call`, `reasoning` and `reasoning-prefilled`
   - `multi-turn` and `long-arguments`
   - `truncated` and `malformed`: these carry an `expected_error` instead of `expected`
   - `regression`: for fixtures taken from bug reports

5. **Validate:** `uv run canitoolcall validate fixtures/acme/`.

6. **Replay it** through every engine you can build, for example `uv run canitoolcall run --engine vllm --family acme`, and look at the result with `uv run canitoolcall matrix`. If the engine has no parser for the family, the adapter's `supports()` must say so, and the result is `unsupported`. When a failure looks like a real engine bug, first check that the fixture is right, then say so in the PR.

7. **Open the PR** with the checklist below.

### PR checklist

- [ ] Every fixture has `provenance` (see below). Every `template_render` fixture names its `generator` and records `template_sha256`.
- [ ] `output_token_ids` and `tokenizer` are present wherever the source allows.
- [ ] Re-running the generator leaves `git diff` clean.
- [ ] `canitoolcall validate` reports 0 issues, and the full lint and test command passes.
- [ ] `docs/formats/acme.md` cites the template and every engine parser by URL at a pinned revision.

## Fixture provenance rules

**Never type a format by hand, and never invent one.** A fixture that looks plausible but is wrong makes an engine look broken when it is not. Every fixture needs a `provenance` block, in this order of preference:

| `kind` | When | `source_url` | `license` |
|---|---|---|---|
| `template_render` | rendered through the official chat template or encoder by your generator script | the template file at the pinned revision | the model repo's license |
| `engine_test` | copied verbatim from an engine's test suite | the test file at a commit, with a line anchor | vLLM, SGLang and transformers: Apache-2.0; llama.cpp and Ollama: MIT. Add `attribution`. |
| `bug_report` | reproduces a public issue | the issue or comment URL | `NOASSERTION`, plus a short quote |
| `recorded` | a real generation captured from the model | the log or dataset | as stated |
| `spec_example` | quoted verbatim from the format's official spec or reference-library docs | the document at a commit, with a line anchor | the spec's license |

**Rendering a past turn is not the same as generating one.** Chat templates serialize *history*, and history can differ from what the model *generates*:

- Mistral v11 templates add a `[CALL_ID]` that the model never emits.
- gpt-oss history puts the recipient in the role header.

Engines parse generations. When the two differ, use an `engine_test` or `recorded` source, and explain why in `notes`.

**Ids never change.** A fixture id is `acme/<kebab-name>` and is stable forever. If a fixture's content changes meaningfully, add a new id and delete the old one.

## Add an engine adapter

An adapter counts only if it runs **the engine's own parser code at a pinned version**. Never reimplement a parser. If something cannot run, report it as `unsupported`.

1. **Build the environment.** `scripts/engines/<engine>.sh` builds `.venvs/<engine>` with pinned versions, plus any checkout or harness under `.engines/<engine>/` (gitignored). It must work on Linux and macOS, on x86_64 and arm64, with no GPU and no model weights.
2. **Write the adapter class.** `src/canitoolcall/adapters/<engine>.py` defines one subclass of `canitoolcall.adapters.base.Adapter`:
   - It must import with the **standard library only** at module level. Import the engine inside methods, because the worker loads the module inside the engine's venv with only `src/` on the path.
   - Implement:
     - `version`, and optionally `commit` and `engine_details`
     - `supports(family, model)`: honest and cheap
     - `parser_config`: record every parser name, template source and sha256, and every flag
     - `units`: return the fixture's `output_token_ids`
     - `parse`
     - `parse_stream`: detokenize each group of ids with **the engine's own** incremental detokenizer, then call its streaming parser the way its server does
   - Optionally implement `special_token_ids`, which enables the `special` strategy.
   - Set `tokens_per_step = "one"` if the engine's server streams one event per generated token; multi-token chunkings are then reported as synthetic instead of counted.
   - Follow the ten rules in the adapter contract in [`docs/DESIGN.md`](docs/DESIGN.md). Each rule exists because an adapter broke without it.
3. **Compiled engines.** For an engine like llama.cpp or Ollama, put the harness in `harnesses/<engine>/`. It speaks JSON lines, and one process serves a whole run.
4. **Register it.** Add one line to the `ENGINES` registry in `src/canitoolcall/adapters/__init__.py`, and add the engine to the matrix in `.github/workflows/nightly.yml`.
5. **Test it.** Put the tests in `tests/adapters/test_<engine>.py`:
   - a fast test of `supports` and `parser_config` that needs no engine
   - `@pytest.mark.engine("<engine>")` tests that replay at least one fixture per supported family through the real worker

Engine exceptions are **outcomes**: catch them and return `ParseResult(exception="Type: msg")`. Raise only for harness problems, which the runner records as `error`.

## Reporting a parser bug upstream

Each failing cell on the matrix site has a page with:

- the failing checks and strategies
- the observed parse, as a diff against the expected one
- the parser configuration
- a one-line replay command

Before you file an issue with an engine:

1. Re-run that command against the pinned version.
2. Check the fixture's provenance.
3. Link the fixture record in the issue.

Please don't file bugs from the untriaged candidates in `docs/DESIGN.md` §6 until they have been turned into fixtures and verified.

## Style

Keep code typed (`mypy --strict`), use dataclasses, and format with `ruff format` (line length 120). Keep dependencies of the main package minimal. Write tests with pytest; the site's tests build their inputs with `RunResults` rather than committing results files.
