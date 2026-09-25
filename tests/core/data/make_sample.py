"""Regenerate the core-test sample fixture from Qwen3's official chat template.

This is TEST DATA for the loader/validator/chunking tests, not part of the
fixture corpus. It is produced exactly as spec/README.md prescribes for
``template_render`` fixtures: render with ``apply_chat_template(tokenize=True)``,
slice off the prompt ids, cut at the first stop id, decode with
``skip_special_tokens=False``.

Run inside an env with transformers>=5 (no torch needed)::

    .venvs/transformers/bin/python tests/core/data/make_sample.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

REPO = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
STOP = "<|im_end|>"
OUT = Path(__file__).parent / "fixtures" / "qwen3-hermes"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "filters": {"type": "object"}},
                "required": ["query"],
            },
        },
    },
]
CALLS = [
    {"name": "get_weather", "arguments": {"city": "Zürich", "unit": "c"}},
    {"name": "search", "arguments": {"query": 'café "best"', "filters": {"tags": ["a", "b"], "max": 3}}},
]
REASONING = "I should call the tools."


def main() -> None:
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    user = [{"role": "user", "content": "Weather in Zürich, and search for the best café?"}]
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": REASONING,
        "tool_calls": [{"type": "function", "function": c} for c in CALLS],
    }

    def ids(messages: list[dict], gen: bool) -> list[int]:
        out = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=gen, tokenize=True)
        return list(out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out)

    prompt = ids(user, True)
    full = ids([*user, assistant], False)
    assert full[: len(prompt)] == prompt, "prompt is not a prefix of the rendered conversation"
    output = full[len(prompt) :]
    stop_id = tok.convert_tokens_to_ids(STOP)
    output = output[: output.index(stop_id)]
    raw = tok.decode(output, skip_special_tokens=False)
    template_sha = hashlib.sha256(tok.chat_template.encode("utf-8")).hexdigest()

    record = {
        "id": "qwen3-hermes/sample-parallel-reasoning",
        "family": "qwen3-hermes",
        "models": [REPO],
        "spec_version": "0.1",
        "provenance": {
            "kind": "template_render",
            "source_url": f"https://huggingface.co/{REPO}/blob/{REVISION}/tokenizer_config.json",
            "revision": REVISION,
            "license": "Apache-2.0",
            "generator": "tests/core/data/make_sample.py",
            "template_sha256": template_sha,
        },
        "tools": TOOLS,
        "raw_output": raw,
        "output_token_ids": output,
        "tokenizer": {"repo": REPO, "revision": REVISION, "mode": "hf"},
        "expected": {"content": None, "reasoning_content": REASONING, "tool_calls": CALLS},
        "tags": ["parallel-calls", "reasoning", "unicode", "nested-json", "string-escapes"],
        "notes": "Core-test sample (history render). Not part of the fixture corpus.",
    }
    family = {
        "slug": "qwen3-hermes",
        "name": "Qwen3 (Hermes-style JSON)",
        "spec_version": "0.1",
        "has_reasoning": True,
        "markers": ["<tool_call>", "</tool_call>", "<think>", "</think>", "<|im_start|>", "<|im_end|>"],
        "reference_models": [
            {
                "repo": REPO,
                "revision": REVISION,
                "variant": "qwen3-hybrid",
                "default_generation_prompt": "<|im_start|>assistant\n",
                "stop_tokens": [STOP],
            }
        ],
        "format_notes": "docs/formats/qwen3-hermes.md",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "family.json").write_text(json.dumps(family, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(raw)


if __name__ == "__main__":
    main()
