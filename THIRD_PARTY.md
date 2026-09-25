# Bundled benchmark sources

This source release includes the benchmark files needed to run the selected
cases. The original license notices are retained with each source tree.

| Bundled path | Source | License |
| --- | --- | --- |
| `bundle/vendor/aider/` | Aider at `5dc9490bb35f9729ef2c95d00a19ccd30c26339c` | Apache 2.0, see `aider/LICENSE.txt` |
| `bundle/vendor/aider/tmp.benchmarks/polyglot-benchmark/` | 26 selected Exercism track exercises from Polyglot dataset revision `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f` | See its `README.md` for track attribution and license links; individual task notices are retained |
| `bundle/vendor/hermesagent20-benchmark/` | HermesAgent-20 at `57d7766bf3db8c40696e3ed937d43c8c85f4cd6c` | MIT, see `hermesagent20-benchmark/LICENSE` |
| `bundle/vendor/terminal-bench-mini/` | Terminal-Bench Mini with the selected Core-19 tasks | See `terminal-bench-mini/LICENSE` |
| `bundle/vendor/tool-eval/` | Tool-Eval 2.1.0 | MIT, see `tool-eval/LICENSE` |
| `bundle/vendor/ifeval/` | Google Research IFEval strict grader | Apache 2.0, see `ifeval/LICENSE` |
| `components/slim_sanitize.py` | Selected EvalPlus 0.3.1 sanitizer functions | Apache 2.0, see `bundle/vendor/evalplus-LICENSE` |

The Hermes verifier build fetches the pinned `nousresearch/hermes-agent`
commit from its public repository when that optional component is selected.
The lightweight default installs only the IFEval, Tool-Eval, and Tree-sitter
dependencies. The selected HumanEval+ source rows and their EvalPlus 0.3.1
reference outputs are included and locked by selector hashes.
