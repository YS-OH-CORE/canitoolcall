# Provider matrix: phase-1 design

Issue: [#27](https://github.com/redd34/canitoolcall/issues/27)

## Decision summary

Start with a **reviewed, PR-based evidence set**, not a submission service.

Phase 1 keeps the existing `probe` network boundary unchanged: `probe` talks only to the
`--base-url` chosen by the user, under the existing key and redirect rules. A separate local
export step turns `probe --json` output into a deliberately smaller **public submission record**.
The contributor reviews that record and opens a normal GitHub PR. CI validates the schema and
rebuilds the provider matrix from committed records.

Phase 1 therefore does **not** add `probe --submit`, a CanIToolCall backend, or any new credential.
The point of phase 1 is to learn whether the public data is useful and governable before adding
an upload path.

A future `--submit` should be considered only after the public schema, moderation policy, and
neutral presentation have survived real PR submissions.

## Ground rules

A provider-matrix record is an observation of one probe run, not an endorsement of a provider,
a benchmark of model quality, or proof that the named provider actually produced the response.

The public record must not contain:

- API keys, authorization headers, cookies, account/project identifiers, or other credentials;
- the submitted `base_url`, query strings, response headers, or raw HTTP bodies;
- arbitrary exception text returned by a provider;
- user prompts or files (the probe uses CanIToolCall's fixed synthetic scenarios);
- per-request latency or a score that invites a provider leaderboard.

The PR itself is public and therefore exposes the contributor's GitHub identity. The contributor
must see the exact public record before deciding to open the PR.

## 1. Consent and scope

### Recommendation

Make publication a **two-step local action**.

1. Run today's probe exactly as documented and save its private/raw report:
   `canitoolcall probe ... --json report.json`.
2. Run a separate export command (name left to implementation) that reads that file and writes
   a sanitized provider-submission JSON. It prints a field-by-field preview and does no network
   I/O.
3. The contributor manually opens a PR containing that file.

The public record should carry a short explicit attestation such as:

> I intentionally ran this CanIToolCall probe against a service I was authorized to use, reviewed
> this sanitized record, and consent to publishing the fields in this file.

This is consent to publish **the sanitized observation**, not permission to publish the raw report
or reuse the contributor's provider credential.

The export must be fail-closed: unknown/raw fields are dropped rather than copied through.

### Rejected: `probe --submit` immediately uploads after probing

This combines "send requests to my provider" and "publish a report to a third party" in one command.
It also changes the current safety boundary of `probe`, which today sends credentials only to the
explicit `--base-url`. A typo, wrapper, or automation could publish when the operator only meant to
test locally.

### Rejected: treat use of `--json` as consent

A local report is not a publication grant. Existing users should not acquire a new network side
effect merely because they already save JSON.

## 2. Storage and hosting

### Recommendation

Store sanitized observations in this repository and accept them through normal PR review.

A possible layout is:

```text
provider-results/
  <provider>/
    <model-slug>/
      <date>-<content-id>.json
```

The exact path is less important than these properties:

- records are immutable evidence; corrections add/supersede rather than silently rewrite history;
- the provider/model identifiers use a small repository-owned normalization table;
- CI validates every record before the matrix can consume it;
- the generated site is derived only from reviewed records on the default branch.

A public record should contain only what the matrix needs, for example:

```json
{
  "schema_version": 1,
  "provider": "groq",
  "model": "example/model",
  "observed_date": "2026-10-07",
  "canitoolcall_version": "x.y.z",
  "scenario_set": "builtin-v1",
  "outcomes": {
    "single-call": {"nonstream": "pass", "stream": "pass", "equivalence": "pass"}
  },
  "attestation": "authorized-run-and-publication-consent"
}
```

Failure details should use bounded repository-defined reason codes where useful. The exporter should
not copy `ProbeOutcome.detail` verbatim because provider/HTTP exception strings are not a stable
public-data boundary.

### Rejected: a new CanIToolCall submission backend

There is no server today. A backend creates an authentication, abuse, retention, privacy, incident
response, uptime, and operating-cost commitment before we know that the matrix is useful.

### Rejected: commit the raw `probe --json` report

The raw report is designed for local diagnosis, not publication. In particular it includes the
`base_url` and free-form details that may reveal local topology or provider-specific identifiers.

## 3. Abuse, fabrication, and rate limits

### Recommendation

Let GitHub provide the phase-1 submission rate limit and identity/accountability layer, and let a
human reviewer decide whether a record enters the corpus.

CI can verify **integrity of the contribution**, not truth of the remote run:

- schema and enum validation;
- known provider/model normalization;
- complete built-in scenario set (no cherry-picked subset);
- version/date sanity checks;
- no forbidden fields such as URLs, headers, raw bodies, or free-form error dumps;
- deterministic regeneration of the matrix;
- duplicate-content detection.

The matrix should label these records **community-submitted observations**. A reviewed PR proves
that a GitHub contribution passed repository checks; it does not prove that the contributor really
used the named provider.

If two accepted runs disagree, keep both and display the cell as mixed/disputed instead of letting
the latest submission erase the earlier one.

### Rejected: accept anonymous POSTs and rate-limit by IP

IP limiting is weak identity, easy to evade, awkward behind NAT/VPNs, and still leaves moderation
and deletion tooling to build. It adds infrastructure without giving trustworthy provenance.

### Rejected: majority vote makes a result "true"

Several submissions can share the same configuration error, model revision, proxy, or commercial
incentive. Counts are useful context, not an authenticity oracle.

## 4. Neutrality and presentation

### Recommendation

Make the public page a **compatibility observation matrix, not a leaderboard**.

- Sort providers and models by stable names, not by pass rate.
- Show scenario/check cells plus observation count and date/version context.
- Do not compute an overall provider score, rank, badge, or "winner".
- Do not rank by latency in phase 1.
- Require the same built-in probe scenario set for comparable submissions.
- Show `mixed` when accepted observations disagree.
- Keep historical observations addressable when provider/model behavior changes.
- Put a visible note above the table: coverage and model revisions differ; results are not a
  provider-quality ranking.

Commercial providers and their employees may submit records under the same schema and review rules.
The matrix should identify the evidence, not infer motives or preferentially weight a provider's own
submission.

### Rejected: one pass percentage per provider

A single percentage hides which scenarios were exercised, makes unequal coverage look comparable,
and turns a structural compatibility tool into a ranking with very little extra effort.

### Rejected: "latest run wins"

Hosted behavior can regress or change by model revision and routing. Replacing history with one
current-looking cell makes disagreements invisible and gives a single submitter disproportionate
control.

## Phase-1 flow

```text
probe --json private-report.json
          |
          | local sanitize + validate (no network)
          v
public-submission.json
          |
          | contributor inspects exact bytes
          v
GitHub PR -> schema/neutrality CI -> human review
          |
          v
reviewed provider-results/ -> deterministic matrix build
```

The original `probe` command keeps all of its current key/redirect/insecure-HTTP guarantees.
Nothing is automatically uploaded.

## Phase-1 acceptance criteria

Before considering an upload service or `probe --submit`, phase 1 should demonstrate:

1. several real provider submissions can be reviewed without maintainers needing private raw logs;
2. the sanitized schema is sufficient to explain matrix cells without leaking endpoint/account data;
3. conflicting reports can coexist without a maintainer manually rewriting history;
4. CI can reject malformed/selective/unsafe records deterministically;
5. the rendered matrix remains useful without scores or rankings.

## What a later `probe --submit` would need

If phase 1 works, a later submission path should reuse the exact same public schema and validation.
It should require an explicit publication confirmation separate from provider probing, identify the
destination before sending, transmit no provider credential, and return a durable receipt that names
the submitted record.

The transport could eventually be a service or a GitHub-assisted flow. That decision should be made
from the observed phase-1 moderation volume rather than assumed up front.

## Non-goals

Phase 1 does not:

- certify provider identity or model provenance;
- benchmark model intelligence, quality, price, uptime, or latency;
- create a CanIToolCall account system;
- accept private/custom endpoints into the public matrix;
- change the existing probe's API-key handling or network destinations;
- automatically file provider bug reports from failures.
