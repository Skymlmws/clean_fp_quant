# Baselines

Each child directory is one self-contained baseline. Implementation code,
launchers, tests, environment setup, and metadata stay together:

```text
baseline_name/
├── manifest.json
├── README.md
├── src/
├── scripts/
└── tests/
```

The harness launches the baseline directory as an external process. Shared
project analysis does not reimplement or import the baseline internals.

Each manifest records:

- which environment variable points to the baseline repository;
- which environment variable selects that baseline's Python interpreter;
- the upstream command to execute;
- how the harness passes its assigned artifact directory.

The manifest records the upstream URL and base revision when applicable, and
every run records the main repository commit containing the baseline code.
