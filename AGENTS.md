# Project Instructions

## Mathematical notation in responses

- Use `$...$` for inline mathematics.
- Use `$$...$$` for display mathematics, with each `$$` delimiter on its own
  line and a blank line before and after the display block.
- Do not use `\(...\)` or `\[...\]`; the project's Markdown renderer may not
  render those delimiters reliably.
- Do not wrap mathematical expressions in ordinary square brackets or place
  rendered formulas inside code fences.
- In lists, leave a blank line before and after every display-math block.
- Before sending a response, verify that all `$` delimiters are paired and all
  LaTeX commands are inside math delimiters.
- For a complex or important formula, also provide an equivalent plain-text or
  code expression so it remains understandable if rendering fails.

Example:

$$
\mathrm{RMS}(X)
=
\sqrt{\frac{1}{NC}\sum_{i=1}^{N}\sum_{j=1}^{C}X_{ij}^{2}}
$$

Plain-text equivalent:

```text
RMS(X) = sqrt(sum(X_ij^2) / (N * C))
```
