# Troubleshooting — When something breaks

> [← Documentation](../README.md)

If your run errored, your score regressed, or your local and remote scores disagree, start here.

| Doc | When to read |
|---|---|
| [common-participant-errors.md](common-participant-errors.md) | Symptom → cause → fix-now → verify, for the most common failures (wrong shape, NaN/Inf, exceeded budget, the 400 ms residual wall-time cap, signature mismatches, import errors, numeric blow-up in deep networks, whole-submission timeout). |
| [faq.md](faq.md) | Quick answers — can I use scipy? what is `residual_wall_time_limit`? how many submissions do I get a day? is there a limit on total evaluation time? why does my submission score worse than my local run? |

## ➡️ Where to look next

- Want the tiered procedure for "estimator runs but something feels wrong"? → [How-to: debugging checklist](../how-to/debugging-checklist.md).
- Suspect drift between machines? → [Reference: `whest doctor`](../reference/whest-doctor.md).
- Need to interpret `per_mlp[i].error`, `budget_exhausted`, `time_exhausted` fields? → [Reference: score report fields](../reference/score-report-fields.md).
- Rules question, an operation you think flopscope is mispricing, or a legitimate need for more residual wall time than the 400 ms cap allows? → email [arc-whestbench@aicrowd.com](mailto:arc-whestbench@aicrowd.com) ([FAQ](faq.md#who-do-i-contact-about-a-rules-question-or-a-flop-that-looks-mispriced)).
