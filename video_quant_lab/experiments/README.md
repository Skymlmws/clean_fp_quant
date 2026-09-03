# Experiments

An experiment selects a baseline manifest and passes arguments in that
baseline's native CLI format. The harness does not translate algorithm options
or import baseline code.

Environment variables in argument strings are expanded when the run plan is
built. This keeps checkpoint and upstream repository paths out of committed
configuration when desired.
