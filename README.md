# CanIToolCall

*caniuse.com for tool calling*: a neutral conformance suite and compatibility matrix for tool-call and reasoning parsers across open-weight model families and inference engines (vLLM, SGLang, llama.cpp, Ollama, HF transformers).

Recorded raw model outputs (fixtures) are replayed **offline** through each engine's *own* parser code, non-streaming and as many chunked streams. No GPU and no model weights are needed.

> Status: pre-release skeleton. The fixture spec is in [`spec/`](spec/README.md), the architecture is in [`docs/DESIGN.md`](docs/DESIGN.md), and per-family format notes are in [`docs/formats/`](docs/formats/README.md).

```sh
uv sync
uv run canitoolcall validate
uv run pytest
```

Licensed under Apache-2.0.
