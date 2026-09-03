# Baselines

Each child directory is a self-contained slot for one upstream baseline. It
contains tracked metadata and an ignored source checkout:

```text
baseline_name/
├── manifest.json
├── README.md
└── source/
```

The harness launches code inside `source/` as an external process. It does not
reimplement the baseline or import its Python internals.

Each manifest records:

- which environment variable points to the baseline repository;
- which environment variable selects that baseline's Python interpreter;
- the upstream command to execute;
- how the harness passes its assigned artifact directory.

`source/` is ignored by the main repository because it retains its own Git
history. The manifest pins the upstream URL and revision, and every run records
the actual checked-out commit.
