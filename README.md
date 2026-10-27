# Contested Quality Suite

A portable, 95-case diagnostic benchmark selected from repeated Qwen3.8 27B and Flash quality runs. It combines seven parent suites: ARC-Challenge, IFEval, HumanEval+, Tool-Eval, Aider Polyglot, HermesAgent, and Terminal-Bench. The selection has 89 contested cases and six separately reported all-fail frontier cases. It is designed for regression diagnosis; its score is not comparable with published parent-suite scores.

**FAST54** is the default, smaller selection: 54 cases from four components. Run `./benchmark` for FAST54, or `./benchmark --all` for the complete 95-case suite. FAST54 is a selected regression panel, not a full-suite score or a standard published benchmark score.

## FAST54 at a glance

| Component | Contested | Frontier | Total | What it checks |
| --- | ---: | ---: | ---: | --- |
| ARC-Challenge | 17 | 1 | 18 | Multiple-choice questions |
| IFEval | 24 | 3 | 27 | Strict instruction following |
| HumanEval+ | 5 | 0 | 5 | Python code against base and extended tests |
| Tool-Eval | 4 | 0 | 4 | Tool-use scenarios TC-72, TC-75, TC-80, and TC-82 |
| **FAST54 total** | **50** | **4** | **54** | |

The other 41 cases are in Aider Polyglot, HermesAgent, and Terminal-Bench; they run with `--all`. “Fast” describes the reduced selection and dependencies, not a guaranteed runtime. Runtime depends on the model, server, hardware, and generation lengths.

## Run

On Linux with Python 3.11 or newer, `uv`, and Bubblewrap (`bwrap`), start an OpenAI-compatible model server at `http://127.0.0.1:8080/v1`, then run FAST54:

```bash
./benchmark
```

That command installs its small Python dependencies on first use. A successful FAST54 run writes `runs/<run-id>/score.partial.json`; the launcher prints the actual run directory. For a different server and a descriptive model label:

```bash
./benchmark --endpoint http://YOUR-SERVER:PORT/v1 --model main --label my-model
```

The launcher discovers the first ID from `/v1/models`; if `--model` is supplied, it must match that first ID. Replace `main` with the ID your server advertises. `./benchmark check` verifies the package without running inference; `./benchmark --help` lists all options.

To run all 95 cases, install Docker and Node.js and use:

```bash
./benchmark --all
```

The full run writes `runs/<run-id>/score.json`. The optional Aider, Hermes, and Terminal cases use bundled tasks and their native agent graders. Their Docker images consume substantially more disk space; see [SIZE_REPORT.md](SIZE_REPORT.md). Model weights and an inference server are not included. Terminal setup checks that Docker has at least 12 GiB free before starting its task images. If `/v1/models` does not advertise context length, pass the model's real capacity with `--terminal-context-length N`.
On NixOS, the Terminal launcher locates the host GCC C++ runtime for Harbor's native Python extensions.

For four concurrent agent tasks in each agent-based component, use `./benchmark --all --aider-threads 4 --hermes-workers 4 --terminal-concurrency 4 --tool-parallel 4`. The text and coding components run their selected questions in order. Only set agent concurrency to a value the model server can support.

`./benchmark doctor` checks the prepared local setup. [AGENTS.md](AGENTS.md) is the short instruction for an agent in this repository.

## FAST54 settings and comparison

The public default is a fixed suite protocol, not an automatic choice of the model publisher's recommended sampler. Its component defaults differ:

| Component | Requested temperature | Output cap | Other requested defaults |
| --- | ---: | ---: | --- |
| ARC | 0 | 256 | top-p 1.0, seed 42 |
| IFEval | 0 | 2,048 | One worker; unspecified sampler fields use server defaults |
| HumanEval+ | 0 | 8,192 | top-p 0.95, top-k 20, min-p 0, seed 0 |
| Tool-Eval | 1.0 | 8,192 per turn | top-p 0.95, seed 1, one worker, at most eight turns |

Check the actual request and server behavior for thinking mode and any fields left to server defaults. In particular, the public ARC driver sends its template setting inside `extra_body` in a direct HTTP request; an OpenAI-compatible server may not interpret that nested field. The public CLI does not offer a switch that applies a model's recommended sampler to every component. A custom runner or sampling overlay is a **different protocol**: retain its exact settings, case IDs, source revision, and scoring changes, and compare it only with runs using the same protocol. Separately maintained instrumented FAST54 wrappers use the same 54 selected cases but add their own instrumentation and may use different sampling and grading; they are not shipped in this repository.

**ARC cutoff limitation:** the public ARC driver caps responses at 256 tokens but does not record the finish reason or token count. It can therefore score a truncated response as an ordinary miss; the default FAST54 output alone cannot prove that ARC responses finished. Preserve or inspect server-side finish telemetry before interpreting those outcomes. The separate instrumented wrapper used for some internal runs records finish reasons and treats cutoffs as non-scorable.

**IFEval cutoff limitation:** the public runner records finish reasons and token counts in `runs/<run-id>/ifeval/requests.jsonl`, but does not automatically exclude length-stopped responses before grading. Inspect that file and treat cutoffs as non-scorable. HumanEval+ and Tool-Eval do guard against truncation; if any response in one of those components is truncated, collection withholds that component's results.

## Protocol v2

[protocol.json](protocol.json) pins the current generation limits and defaults. HumanEval+, Terminal, and Tool-Eval request up to 8,192 output tokens; Aider defaults to one concurrent task for one-session endpoints. HumanEval+ records each response's finish reason and token count. Tool-Eval records each turn's finish reason and token count. A length-stopped HumanEval+ or Tool-Eval response is marked non-scorable. Terminal setup exceptions are also non-scorable. These changes affect outcomes, so v2 scores must not be compared as if they used the v1.0.1 generation protocol. Run metadata and score files carry the protocol hash. Protocol 2.1.0 also corrects some TC-80 negation false failures; its text heuristic can still require manual review. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Scoring and contents

The full primary score is the equal-weight mean of the seven contested component pass rates. The six frontier cases are reported separately. Aider and Terminal use native pass@2; IFEval uses the official strict prompt grader; Hermes and Tool-Eval use native scenario pass; HumanEval+ requires both base and plus tests to pass. FAST54 produces a partial score with its component coverage and completed-component macro; it does **not** produce the seven-component primary score. Its four frontier cases remain separate. Report incomplete or non-scorable cases instead of treating them as model failures.

[suite.json](suite.json) holds selected case IDs, tasks, grading specifications, and aggregate historical pass/fail counts. It intentionally omits historical run paths and per-arm outcomes. [selectors/](selectors/) contains the fixed inputs. [bundle/vendor/](bundle/vendor/) contains the needed task and scorer source, with SHA-256 pins in [vendor-lock.json](bundle/vendor-lock.json). [THIRD_PARTY.md](THIRD_PARTY.md) identifies upstream material and licenses.

The selection came from 42 catalogued suite-run entries and 8,972 question-level observations. The original private run logs are not in this repository. Unanimous easy cases and invalid observations were excluded; disputed and repeat-flipping cases were prioritized. The selected cases were checked for prompt, answer-key, and grader issues. Different historical quantizations and serving settings can affect outcomes, so disagreement identifies sensitive questions rather than a single cause. No new model was run on this hotlist during its construction.
