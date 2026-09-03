# Analysis ownership

This package contains project-owned analysis code that is independent of a
quantization algorithm. Wan-specific tensor layout knowledge is allowed here;
imports from a baseline's quantizer or transform implementation are not.

The `wan/` package currently owns:

- activation statistics;
- token-by-channel matrix sampling;
- structured outlier detection;
- activation artifact paths and disk capture;
- analysis-side names for Wan linear sites.

Baseline-specific analysis should remain with that baseline. For example, a
weight plot that directly constructs FP-Quant's `Quantizer` is not moved into
this package.
