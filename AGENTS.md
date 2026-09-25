# Contested quality suite

Run `./benchmark` from this directory for the four lightweight components
(54 cases). Run `./benchmark --all` for all seven components and 95 cases;
the Aider, Hermes, and Terminal components require Docker and larger native
runtimes. The command discovers the first model at `http://127.0.0.1:8080/v1`.
Use `--endpoint URL/v1` when the model server is elsewhere. Report the run
directory and the score path printed by the command. Keep the selected cases
and grader thresholds fixed.
If the endpoint omits context length, add `--terminal-context-length N` using
the server's actual context capacity. Do not mix results from different
`protocol.json` hashes.
