# Baseline command manifests

Files in this directory describe how to launch an upstream baseline. They do
not reimplement the baseline and should not import its Python internals.

Each manifest records:

- which environment variable points to the baseline repository;
- which environment variable selects that baseline's Python interpreter;
- the upstream command to execute;
- how the harness passes its assigned artifact directory.

An external baseline may live anywhere on disk. Pin its upstream Git commit in
the experiment record instead of copying its source into this repository.
