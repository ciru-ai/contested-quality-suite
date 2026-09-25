# Contested Quality Suite

A portable, 95-case diagnostic benchmark selected from repeated Qwen3.8 27B and Flash quality runs. It combines seven parent suites: ARC-Challenge, IFEval, HumanEval+, Tool-Eval, Aider Polyglot, HermesAgent, and Terminal-Bench. The selection has 89 contested cases and six separately reported all-fail frontier cases. It is designed for regression diagnosis; its score is not comparable with published parent-suite scores.

## Run

On Linux with Python 3.11 or newer, `uv`, and Bubblewrap (`bwrap`), start an OpenAI-compatible model server at `http://127.0.0.1:8080/v1`, then run:

```bash
./benchmark
```

That command installs its small Python dependencies on first use, runs the 54 cases in ARC, IFEval, HumanEval+, and Tool-Eval, and writes `runs/<run-id>/score.partial.json`. To run all 95 cases, install Docker and Node.js and use:

```bash
./benchmark --all
```

The full run writes `runs/<run-id>/score.json`. The optional Aider, Hermes, and Terminal cases use bundled tasks and their native agent graders. Their Docker images consume substantially more disk space; see [SIZE_REPORT.md](SIZE_REPORT.md). Model weights and an inference server are not included. Terminal setup checks that Docker has at least 12 GiB free before starting its task images. If `/v1/models` does not advertise context length, pass the model's real capacity with `--terminal-context-length N`.
On NixOS, the Terminal launcher locates the host GCC C++ runtime for Harbor's native Python extensions.

For four concurrent agent tasks in each agent-based component, use `./benchmark --all --aider-threads 4 --hermes-workers 4 --terminal-concurrency 4 --tool-parallel 4`. The text and coding components run their selected questions in order. Only set agent concurrency to a value the model server can support.

For a different server, use `./benchmark --endpoint https://your-server.example/v1`. Use `--label NAME` to record the checkpoint and quantization. `./benchmark check` verifies the locked package inputs without installing or running anything. `./benchmark doctor` checks the prepared local setup. Run `./benchmark --help` for component selection and other options. [AGENTS.md](AGENTS.md) is the short instruction for an agent in this repository.

## Protocol v2

[protocol.json](protocol.json) pins the current generation limits and defaults. HumanEval+, Terminal, and Tool-Eval now request up to 8,192 output tokens; Aider defaults to one concurrent task for one-session endpoints. HumanEval+ records each response's finish reason and token count. Tool-Eval records each turn's finish reason and token count. A length-stopped HumanEval+ or Tool-Eval response is marked non-scorable. Terminal setup exceptions are also non-scorable. These changes affect outcomes, so v2 scores must not be compared as if they used the v1.0.1 generation protocol. Run metadata and score files carry the protocol hash.

## Scoring and contents

The full primary score is the equal-weight mean of the seven contested component pass rates. The six frontier cases are reported separately. Aider and Terminal use native pass@2; IFEval uses the official strict prompt grader; Hermes and Tool-Eval use native scenario pass; HumanEval+ requires both base and plus tests to pass. A partial run reports its coverage and completed-component macro score rather than a full-suite primary score.

[suite.json](suite.json) holds selected case IDs, tasks, grading specifications, and aggregate historical pass/fail counts. It intentionally omits historical run paths and per-arm outcomes. [selectors/](selectors/) contains the fixed inputs. [bundle/vendor/](bundle/vendor/) contains the needed task and scorer source, with SHA-256 pins in [vendor-lock.json](bundle/vendor-lock.json). [THIRD_PARTY.md](THIRD_PARTY.md) identifies upstream material and licenses.

The selection came from 42 catalogued suite-run entries and 8,972 question-level observations. The original private run logs are not in this repository. Unanimous easy cases and invalid observations were excluded; disputed and repeat-flipping cases were prioritized. The selected cases were checked for prompt, answer-key, and grader issues. Different historical quantizations and serving settings can affect outcomes, so disagreement identifies sensitive questions rather than a single cause. No new model was run on this hotlist during its construction.
