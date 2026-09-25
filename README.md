# CanIToolCall

**caniuse.com for tool calling.** CanIToolCall is a neutral conformance suite and compatibility matrix for the tool-call and reasoning parsers of open-weight models, across inference engines: vLLM, SGLang, llama.cpp, Ollama and HF transformers.

The same model can call tools correctly on one server and break on another. Each engine has its own parser that turns the model's raw output into `tool_calls`, `reasoning_content` and `content`, and these parsers break often: arguments get dropped, markers leak into content, and streaming gives a different answer from non-streaming. CanIToolCall replays **recorded raw model outputs** (fixtures) through each engine's **own parser code**, offline, with no GPU and no model weights. Each output is parsed once without streaming and once for each way of splitting the stream into chunks. Every failure it finds comes with a fixture that can be pasted into a regression test.

> **Status: pre-release (0.1.0.dev0).** Nothing has been published to PyPI yet. Until it is, run the commands below from a checkout with `uv run canitoolcall …` in place of `uvx canitoolcall …`.

**The matrix:** `https://<owner>.github.io/canitoolcall/`. This is a placeholder: the nightly workflow publishes the site once the repository is public.

## 60-second quickstart

**Check your own stack.** Point the live probe at any OpenAI-compatible endpoint (vLLM, SGLang, llama-server, Ollama, LM Studio or a hosted API):

```sh
uvx canitoolcall probe --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B
```

It sends a short series of scripted tool-use requests, first without streaming and then with streaming, and reports pass or fail for each one:

- a single call and parallel calls
- a plain answer where no call is needed
- nested and unicode arguments, and a tool with no arguments
- a forced `tool_choice`
- a follow-up turn after a tool result
- reasoning followed by a call

The API key is read from `$OPENAI_API_KEY`; use `--api-key-env` to name another variable. The key is only sent in the `Authorization` header and is never logged. Add `--json report.json` to keep a machine-readable report. The exit code is `0` when everything passes, `1` on failures and `2` on usage errors.

**Replay the offline suite against an engine.** This needs a checkout, because each engine runs in its own isolated environment:

```sh
git clone https://github.com/<owner>/canitoolcall && cd canitoolcall
uv sync
bash scripts/engines/vllm.sh           # builds .venvs/vllm (pinned; CPU only; no weights)
uv run canitoolcall run --engine vllm   # writes results/vllm-<version>.json
uv run canitoolcall matrix              # renders site/_build/index.html from results/*.json
```

Other useful commands:

- `canitoolcall engines` lists the engine adapters and the interpreter each one would use.
- `canitoolcall validate` checks fixtures against the spec.
- `canitoolcall matrix --results results/2026-09-25` renders the committed snapshot of real runs through all five engines. [`results/2026-09-25/README.md`](results/2026-09-25/README.md) explains the snapshot, and its `triage.jsonl` classifies every failure, with a repro command for each.

## What it checks

Every fixture is parsed once without streaming and once for each **chunking strategy**. A strategy splits the output into stream deltas by grouping its **token ids**, never its characters, because engines never split a token. The default strategies are:

- `one`: the whole output as a single delta
- `special`: split at special-token boundaries
- `token`: one token per delta
- `rand:1:8` … `rand:5:8`: five seeded random groupings

Each strategy is seeded, so it produces the same deltas on every machine.

| Check | Passes when |
|---|---|
| `expected_match` | the parse equals the fixture's expected content, reasoning and tool calls (arguments compared as parsed JSON) |
| `expected_error` | truncated or malformed output fails gracefully, in one of the accepted ways |
| `stream_equals_nonstream` | every streamed result equals the non-streaming one |
| `split_invariance` | the result does not depend on where the stream was split |
| `no_leakage` | no format marker (`<tool_call>`, `<|call|>`, `[TOOL_CALLS]`, …) leaks into content, reasoning or arguments |
| `arguments_json` | every `arguments` string is a JSON object |
| `arguments_schema` | the arguments validate against the tool's JSON Schema, and the name is one of the offered tools |
| `parallel_order` | parallel calls come back complete and in order |

A case gets the worst status over its checks: `fail` > `error` > `soft pass` > `pass`. Differences only in whitespace, such as a leading `\n` in the reasoning, count as `soft pass`, and the matrix shows them separately. `unsupported` means the engine version has no parser for that model, and CanIToolCall never guesses in its place. The exact rules are in [`spec/README.md`](spec/README.md).

### Families and engines

- **Model families:** `qwen3-hermes`, `qwen3-xml`, `gpt-oss`, `deepseek`, `kimi`, `glm`, `llama`, `mistral` and `gemma4`. Each has format notes with their sources in [`docs/formats/`](docs/formats/README.md).
- **Engines (pinned):**

  | Engine | Version |
  |---|---|
  | vLLM | 0.30.0 |
  | SGLang | 0.5.20 |
  | llama.cpp | `a25c9865` |
  | HF transformers | 5.17.0 |
  | Ollama | `7af39318` (stretch goal) |

- **How adapters run the parsers:**
  - Python engines: the adapter imports the engine's parser classes and calls them the way the engine's own server does.
  - llama.cpp and Ollama: the adapter drives a small compiled harness that links the engine's parser code.

  The adapter contract is in [`docs/DESIGN.md`](docs/DESIGN.md).

Every fixture records where it came from:

- a render through the model's official chat template or encoder, reproduced by a script in `scripts/fixtures/`,
- a case copied from an engine's test suite, with its license and a line-anchored URL,
- or a public bug report.

Formats are never typed by hand.

## For engine maintainers: vendor the fixtures

The fixtures are language-neutral JSON Lines files: see [`fixtures/`](fixtures/) and the schemas in [`spec/`](spec/). The wheel ships both of them, so you can use the suite without cloning this repository.

**From pytest.** The package registers a pytest plugin that stays inert until a test asks for it:

```python
# tests/test_canitoolcall.py in your engine's repo
from canitoolcall.pytest_plugin import assert_conforms

def test_conformance(canitoolcall_fixture):          # one test per fixture, ids = fixture ids
    result = my_engine_parse(canitoolcall_fixture)    # -> canitoolcall.results.ParseResult
    assert_conforms(canitoolcall_fixture, result)
```

```sh
pip install canitoolcall
pytest --canitoolcall-family qwen3-hermes --canitoolcall-family glm   # optionally --canitoolcall-tag parallel-calls
```

Use `--canitoolcall-fixtures PATH` to test against your own copy of the corpus.

**From any language.** Read `fixtures/<family>/*.jsonl`. Use `output_token_ids` when a fixture has them, rather than re-encoding `raw_output`: re-encoding is lossy for some tokenizers. Stream the ids through your own detokenizer and parser, and compare with `expected`. The fields are documented in [`spec/README.md`](spec/README.md).

**Reproduce a failure from the matrix.** Every failing cell has a page with:

- the failing checks and strategies
- the observed parse, as a diff against the expected one
- the exact parser configuration
- a one-line command that replays just that fixture file

## Contributing

Adding a model family takes a single PR; see [CONTRIBUTING.md](CONTRIBUTING.md). Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first. To report a security problem, see [SECURITY.md](SECURITY.md).

## License

Apache-2.0; see [LICENSE](LICENSE). Some fixtures come from engine test suites under their own licenses (Apache-2.0 or MIT). The license is recorded in each fixture's `provenance`.
