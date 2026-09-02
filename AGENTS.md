# Project Instructions

## CLI formula display

Responses are displayed in a terminal without TeX rendering.

- Never output LaTeX delimiters: `$`, `$$`, `\(`, `\)`, `\[`, or `\]`.
- Never use LaTeX commands such as `\frac`, `\sqrt`, `\sum`, `\mathrm`, or `\mathbb`.
- Convert every formula to Unicode or ASCII plain text.
- Put complicated formulas in plain-text code blocks.
- Before sending a response, scan it and rewrite any remaining LaTeX.
- Only create LaTeX in a separate file when the user explicitly requests it.

Example:

Bad:
$$ \sqrt{\frac{1+1+1+100}{4}} \approx 5.07 $$

Good:
RMS = sqrt((1 + 1 + 1 + 100) / 4) ≈ 5.07