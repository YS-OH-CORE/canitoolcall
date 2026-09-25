# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Fixture spec v0.1 (`spec/`), 469 fixtures across nine model families, each with provenance.
- Offline replay through the pinned parsers of vLLM 0.30.0, SGLang 0.5.20, llama.cpp `a25c9865`,
  Ollama `7af39318` and HF transformers 5.17.0, with eight checks and seeded token-level chunking.
- Per-engine streaming granularity: multi-token chunkings are reported as synthetic for engines
  whose servers stream one token per event (`run.synthetic_strategies`).
- `canitoolcall probe` for live OpenAI-compatible endpoints, a pytest plugin, and the static matrix site.
- A committed results snapshot (`results/2026-09-25/`) with every failure triaged.
