# Security policy

## Supported versions

CanIToolCall is pre-release. Only the latest release on PyPI, and the `main` branch, receive fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Include the version or commit, the command you ran, and what happened. We aim to acknowledge reports within a week and to agree a disclosure date with you once a fix is ready.

## Scope

In scope:

- **`canitoolcall probe`.** It sends your API key (from `$CANITOOLCALL_API_KEY` or the variable named by `--api-key-env`) to the `--base-url` you give it. The key must only ever go in the `Authorization` header of requests to that URL: never to another host (redirects are not followed), never over plain http to a non-loopback host without `--allow-insecure`, and never in logs, reports, JSON output or error messages. Any leak of the key is a vulnerability.
- **The matrix site generator.** It renders data from results files, including raw model outputs and engine error messages, into HTML. Any way to inject script or markup into the generated site is a vulnerability.
- **The CI workflows.** They use no secrets, and the release workflow publishes with PyPI trusted publishing. Any way for a pull request to obtain write tokens, or to publish, is a vulnerability.
- **Fixture loading.** Fixtures are data. Code execution from a fixture or `family.json` file is a vulnerability. Tokenizer pins must name the family's reference model (or its declared mirror) at a full commit sha, and only the reviewed pins in `canitoolcall.adapters.base.REMOTE_CODE_ALLOWLIST` may load repository code (`trust_remote_code`).

Out of scope:

- **Bugs in the inference engines.** Bugs in vLLM, SGLang, llama.cpp, Ollama or transformers go to those projects. If a parser bug has security impact, report it to that engine's security contact.
- **Engine setup scripts.** `scripts/engines/*.sh` download and run pinned third-party engine code in isolated local environments by design.
