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

## Shared GPU utilization

This machine is a shared laboratory server. Make efficient use of GPUs
legitimately available to the current task when doing so is expected to provide
a meaningful improvement in completion time or throughput.

- Explicit user constraints on GPU IDs, GPU count, memory limits, or concurrency
  always take precedence.
- Respect scheduler allocations, `CUDA_VISIBLE_DEVICES`, GPU reservations, and
  laboratory resource policies. Do not use a GPU outside the resources assigned
  or otherwise made available to the current task merely because it appears
  idle.
- Before starting substantial GPU work, inspect GPU utilization, free memory,
  and running processes. If availability is ambiguous, observe utilization over
  a short interval rather than relying on a single snapshot.
- Prefer parallel execution across multiple available GPUs when the workload can
  be safely partitioned and parallelism is expected to provide a meaningful
  speedup.
- When several suitable GPUs are genuinely available, favor using enough of them
  to keep the workload moving efficiently instead of leaving resources unused
  by default.
- Never interrupt, replace, preempt, oversubscribe, or materially slow down
  workloads belonging to other users.
- Treat a GPU with an unknown or unrelated existing process, or substantial
  memory allocation, as occupied unless sharing is clearly safe and permitted.
- Use separate devices, output directories, logs, communication ports, and
  temporary files for parallel workers to prevent collisions. Ensure partial
  results can be identified, resumed, and combined correctly.
- Avoid parallelization when startup overhead, model duplication, I/O
  contention, memory pressure, or coordination cost would outweigh the expected
  benefit.
- If the safe or efficient degree of parallelism is uncertain, start
  conservatively, measure resource usage and performance, and scale out onto
  additional available GPUs.
- If resource contention appears, reduce parallelism when feasible and follow
  the laboratory's priority and fairness conventions.
- Release GPU resources promptly when a worker finishes or fails. Clean up only
  orphaned worker processes created by the current task.
