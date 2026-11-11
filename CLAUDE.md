# Instructions for Claude in this repository

## Model routing: Opus for hard thinking, Sonnet for well-defined work (user, 2026-11-11)

Use **Opus** (the main session) only for work that needs judgment. Hand **well-defined**
work to **Sonnet** subagents (`Agent` tool, `model: "sonnet"`). Reports especially go to
Sonnet.

**Opus keeps:**

- planning, design and architecture: the plan, decisions D*/O*, gate rules, and which
  amendments the results justify;
- diagnosing failures or surprising results (why a run over-fits, why a test fails in a
  non-obvious way);
- code where correctness is subtle: causality and leakage, the training loop, losses,
  fusion models, anything touching D15 (the test seal);
- reviewing Sonnet's output before it is reported to the user, and the final reply.

**Sonnet gets:**

- reports, summaries, write-ups and doc updates whose content is already decided (B6
  report briefs, PROJECT_LOG entries from given numbers, runbook edits);
- self-contained modules with a precise spec in the plan: metrics and statistics,
  calibration formulas, baselines, plotting, CLI wiring, SLURM scripts;
- tests for a given spec, mechanical refactors, extracting numbers from result files.

**Calibrating the choice.** Default to Sonnet when all of these hold:

1. the inputs, outputs and acceptance test can be written down in the prompt;
2. a wrong answer would show up in a test or a quick review;
3. no project decision has to be made.

Keep the task on Opus if any of these fails, or if the task touches causality, leakage, the
test seal or the loss.

When work splits, Opus writes the spec, Sonnet implements it in disjoint files, and Opus
reviews and integrates. Sonnet prompts must be self-contained: the files to read, the
output path, the acceptance checks, and the standing rules:

- no commits;
- no heavy jobs on the laptop;
- use `D:\RULModel\.venv\Scripts\python.exe`;
- never unseal test results.

Escalate a Sonnet result to Opus when it fails its checks twice or raises a design
question.
