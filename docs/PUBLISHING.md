# Publishing checklist (human steps)

Nothing is pushed or published from the development machine. The repository, the PyPI project and the Pages site are all created by a maintainer, by hand, following the steps below. Do them in order.

## 1. Before the first push

- [ ] Check that the names are still free:
  - PyPI: `https://pypi.org/pypi/canitoolcall/json` returns 404.
  - GitHub: the org or repo `canitoolcall` is available.
- [ ] Replace the placeholders:
  - `<owner>` in `README.md`: the clone URL and the matrix URL.
  - `[project.urls]` in `pyproject.toml`.
  - The contact method in `CODE_OF_CONDUCT.md` (search for `CONTACT METHOD`).
  - The stub notice in `SECURITY.md`.
- [ ] Run the full local check, and confirm the build is clean:

  ```sh
  uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy && uv run pytest
  uv run canitoolcall validate
  uv build && uvx twine check --strict dist/*
  ```

- [ ] Audit the fixtures before release. Every fixture must have provenance. Re-run every `scripts/fixtures/*/build.py`, and check that `git diff` stays clean.
- [ ] Make sure no results file or built site is committed (`/results/` and `/site/_build/` are gitignored), and that `.spikes/`, `.engines/` and `.venvs/` are ignored too.

## 2. GitHub repository

- [ ] Create the repository (public) and push `main`.
- [ ] Under **Settings → Pages**, set **Source** to **GitHub Actions**.
- [ ] Under **Settings → Code security**, enable **Private vulnerability reporting**, which `SECURITY.md` relies on.
- [ ] Under **Settings → Environments**:
  - The `github-pages` environment is created automatically on the first deploy.
  - Create an environment named `pypi`, and add required reviewers so a release needs an approval.
- [ ] Protect `main`: require the `ci` checks to pass before merging.
- [ ] Run **Actions → nightly → Run workflow** once by hand. Check that:
  - each engine job either uploads a results file or shows a "not run" notice;
  - the site builds;
  - the Pages URL works.

  Then put that URL into `README.md` in place of the matrix link placeholder.

## 3. PyPI (trusted publishing, no API token)

- [ ] On PyPI, go to **Your account → Publishing → Add a new pending publisher** and fill in:

  | Field | Value |
  |---|---|
  | PyPI project name | `canitoolcall` |
  | Owner | the GitHub org or user |
  | Repository | `canitoolcall` |
  | Workflow | `release.yml` |
  | Environment | `pypi` |

- [ ] Set `__version__` in `src/canitoolcall/__init__.py`, for example `0.1.0`, and commit.
- [ ] Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
- [ ] Go to **Actions → release → Run workflow**, choose the tag `v0.1.0` in "Use workflow from", and approve the `pypi` environment. The workflow refuses to run on anything but a `v*` tag, and fails if the tag does not match the built version.
- [ ] Check the release from a clean machine: `uvx canitoolcall --version` and `uvx canitoolcall probe --help`.

## 4. After release

- [ ] Announce the matrix to engine maintainers, and offer the pytest plugin for their CI (see the README).
- [ ] Before filing any upstream issue, turn the untriaged candidates in `docs/DESIGN.md` §6 into fixtures and verify them.
