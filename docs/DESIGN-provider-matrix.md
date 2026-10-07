# Proposal: a reviewed provider-observation pilot

**Status: proposed, not implemented or approved.** This answers [#27](https://github.com/redd34/canitoolcall/issues/27).
The decision requested is whether to prepare a small, manually reviewed report format, not permission to build a
submission service. `probe --submit` does not exist today and this proposal does not add it.

## Recommendation and alternatives

Start with an opt-in, PR-reviewed catalogue of dated observations, separate from the offline engine matrix. Keep the
ordinary `probe` command local-output-only and unchanged. No upload backend, automatic provider reruns, or leaderboard.

| Question | Recommended first step | Alternative not chosen, and trade-off |
|---|---|---|
| Consent and scope | Publish only a previewed, explicitly approved derived record. Probing and publishing are separate decisions. | Upload the existing JSON with a one-time opt-in flag: convenient, but it crosses a new disclosure boundary and can disclose more than scores. |
| Storage and hosting | Small data-only PRs in this repository, reviewed before a static catalogue includes them. | New ingestion API/database: easier at scale, but adds authentication, moderation, storage and availability obligations before demand is established. |
| Abuse and rate limits | Treat reports as untrusted claims; bounded static validation, manual intake and no CI calls to report endpoints. | Automatically rerun each submitted URL: can spend money, hit private services or turn submissions into requests against third parties. |
| Neutrality | Show dated, per-scenario observations and disagreements within explicit comparison groups. | Rank providers by a pooled pass rate: unlike coverage, request conditions and selected submissions are not interchangeable. |

The cost is slower intake and weaker evidence than controlled independent measurement. That is preferable to presenting
an easy-to-game upload count as confidence. A reasonable **proposed pilot cap**, not a measured capacity, is ten accepted
reports, one report per PR, each at most 32 KiB. Review the workload and usefulness before increasing it.

## Why the existing report is not the public format

Source basis: [`probe.py`](../src/canitoolcall/probe.py) and [`results.py`](../src/canitoolcall/results.py) at
[`bdade62a`](https://github.com/redd34/canitoolcall/commit/bdade62a9513ccd8657b5e153a9f7ff2421c832d).
`ProbeReport.to_dict()` includes `base_url`, `model`, exact timestamps and outcomes. Each outcome can include free-form
`detail`, `warnings` and `observed`; the latter contains response content, reasoning and tool-call arguments. `safe_url()`
removes userinfo/query/fragment, not the path. A hostname, deployment name or response can itself be private. These are
useful local diagnostics, not inherently safe public data. This is a prospective publication-design constraint, **not a
claim that the current local-report feature uploads data or that a real credential leak was observed**.

The live probe also measures the model/template/sampling/server/parser stack, not just a parser replaying known text.
Its nine built-in scenarios normally generate eighteen requests and twenty-seven outcome rows: the nine mode-comparison
rows reuse the responses, rather than adding nine independent trials. Warnings do not change status; an errored request
makes its comparison row `skip`. Keep these distinctions instead of borrowing the offline matrix's interpretation.

## Publication consent and data minimization

The reporter must be authorized to test the chosen account/endpoint and to publish this observation. Running `probe`,
saving `--json`, or agreeing to a provider's API terms is not consent to publish. Before the **first public push, PR or
attachment**, review the exact bytes to be shared. Review after opening a public PR is too late to prevent disclosure.
Consent should cover the provider/model identifiers, dated outcomes, submitting GitHub identity, public hosting and
possible copies/forks; it does not authorize another probe, recurring submission or use of the reporter's credentials.
GitHub describes the limitations of [removing already exposed data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

Use a positive allowlist for a new, versioned public-record schema. Do not accept the raw `ProbeReport` and try to redact
arbitrary text afterwards. Proposed fields, with bounded lengths and no extra properties:

| Field group | Public record proposal |
|---|---|
| Identity and time | Schema version; reporter-chosen run ID; UTC observation date; approved public provider ID and exact requested public model ID. No internal deployment aliases. |
| Reproduction context | CanIToolCall version and source revision; built-in scenario-set revision/digest; selected modes; effective timeout/concurrency and request-setting profile. These settings are **not all in today's report** and need explicit capture in a later exporter. |
| Route uncertainty | Declared public routing profile where available; resolved model revision only when actually known. Otherwise `unknown`, not an inferred backend or engine version. |
| Outcomes | Complete unique keys `(scenario_id, check, stream)` with `pass/fail/error/skip` and warning counts; summaries recomputed from the rows. |
| Review record | Review scope/date, source PR and correction/supersession links. Keep content approval separate from any independently reproduced result. |

Exclude the raw URL, endpoint path/IP, headers, keys and environment-variable values, prompts, responses/reasoning,
arguments, free-form diagnostics and latency from this pilot. Do not publish hashes of the excluded private text as a
redaction substitute. Public model IDs and route labels still need review; an allowlisted string field is not itself a
privacy guarantee. Keep detailed evidence locally, and request separately consented, minimized evidence only when useful.
If the observation cannot be described without disclosing private identifiers, do not publish it in this pilot.

For future diagnostics use reviewed finite reason codes, never text copied from an error. Do not guess HTTP/auth/rate-limit
causes from today's free-form strings or silently recategorize the recorded status. A restricted summary reduces
reproducibility, so its evidence label must acknowledge that limitation.

## Storage, review and corrections

After approval of this design, a separate implementation PR could propose `provider-observations/` plus a versioned
schema and trusted validator. Keep it separate from `results/`, whose records feed the existing parser matrix. Initial
presentation can be a small static index of reviewed records, not a new service or changes to the existing matrix.
These paths and components are proposals; none are added here.

A data PR is visible before merge. It must contain only the approved derived record and the reporter's consent statement,
not private raw evidence. Review is a decision about format, scope and publication, not proof the provider executed the
claimed requests. A revision/digest binds bytes but does not authenticate a remote model run. Label observations
`self-reported`; separately corroborated runs should have their own records, dates and explicit matching conditions.
Do not turn ten submissions by one actor or copies of one run into ten independent confirmations.

Corrections should link to and supersede the erroneous observation, with the static index following the corrected state.
Ordinary correction does not justify silently changing a measurement. Sensitive-data incidents instead follow
[SECURITY.md](../SECURITY.md) and GitHub removal procedures; retaining leaked evidence for history is not the policy.
No promise can be made to erase copies already obtained by others.

## Abuse controls without a provider-calling backend

A later data validator should require one bounded UTF-8 JSON object, reject duplicate JSON keys/non-finite numbers and
unknown fields, constrain identifiers/enumerations, and validate the exact expected outcome inventory for the pinned
scenario profile. Missing/duplicate rows or mismatched totals are invalid reports, not passing comparisons. An explicit
`error` or `skip` is valid evidence of that outcome, not a missing field to fill with `pass`.

Validate data with trusted code and schema from the reviewed base, not scripts or workflows supplied by the report PR.
Use an unprivileged, isolated job, no secrets or provider keys, and no fetching URLs named in a report. Escape text in any
static rendering; a model/provider identifier is data, not HTML, a shell fragment or a path to execute. Do not introduce a
privileged workflow that executes a fork's code; see GitHub's
[`pull_request_target` guidance](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target).

Proposed manual intake limits: one open report PR per contributor and one accepted observation per contributor/comparison
group/day, retaining all outcomes from that run. These are moderation limits, not protection against multiple accounts.
Use existing repository moderation, pause intake when review capacity is exceeded, and record counterevidence rather
than demanding unlimited reruns. Affiliation and repeat-report patterns inform review but are not truth scores.
A malicious reporter can still fabricate a plausible JSON object. The catalogue must say so.

No CI job should call a hosted provider to approve a submission. Any independent rerun is a separate, explicit action by
an authorized tester using their own account and bounded request/time/cost budget. Preserve the plain probe's key-source,
explicit destination, no-redirect and insecure-key-transport rules verbatim; do not make publishing implicitly set
`--allow-insecure`, discover credentials or retry against another provider. For this pilot, accepting only approved
public HTTPS provider routes is a *narrower publication scope*, not a change to ordinary local probing.

## Neutral presentation

Keep provider observations on a separate page titled as dated **end-to-end tool-use observations**. A failing live probe
alone does not locate a parser defect or establish that the provider intentionally removed support.

Group only records with matching provider/routing profile, requested model, client/scenario revisions, modes and effective
request settings. Show unknown backend revisions explicitly; matching aliases do not prove identical deployments. Region
or account conditions that cannot be safely disclosed remain a comparison limitation, not grounds to assume equivalence.
Do not combine different profiles into one cell or infer an offline engine's version from its provider name.

Show the individual scenarios, coverage denominator, errors, skips and warnings. Keep mode equivalence separate from
request outcomes and avoid an overall provider score or speed ranking. Missing reports mean **not observed**, not failure.
Conflicting reports stay visible. A newer observation can be more relevant without making an older observation false.
As an initial display policy, mark observations older than thirty days as historical; this is a reviewable presentation
choice, not a measured shelf life for providers. Record dates always remain visible.

Opt-in reports have selection and affiliation bias, even after validation. Neutral presentation makes those limits visible;
it does not make the sample representative. Reconsider a public matrix only if the pilot produces useful, independently
checkable observations and maintainers accept the ongoing moderation cost.

## Decision and later acceptance gates

For #27, maintainers can approve the bounded pilot, request changes or decline it. Until that decision, do not implement
an exporter, `--submit`, ingestion, hosting or automated probes.

Before accepting real reports in a later implementation, demonstrate:

1. Private synthetic values placed in URL/model/response/detail/warning fields never enter the derived public record;
   unknown identifiers are rejected for review, not copied through. The publication preview precedes public Git writes.
2. Invalid inventories, changed payloads under a reused run ID and duplicate JSON keys are rejected; identical submissions
   deduplicate without converting repeated requests into independent evidence.
3. All validation/rendering tests use local synthetic data; report URLs cannot cause network calls or privileged execution.
4. Errors/skips/warnings, conflicting results, unknown revisions and historical observations remain distinguishable.
5. Ordinary `probe` behavior and existing safety tests remain unchanged, and the maintainer approves the pilot's intake,
   consent, schema and display limits before any real submission is solicited.
