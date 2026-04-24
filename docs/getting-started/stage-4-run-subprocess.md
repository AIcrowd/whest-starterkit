# Stage 4: Subprocess Runner

Stage 3 runs in your interpreter. Stage 4 spawns each estimator call in a fresh subprocess — the same isolation the grader uses. Catches:

- Shared global state between calls
- Stale RNG seeds from previous calls
- Memory leaks
- Imports that fail in a clean process

## Run it

```bash
whest run --estimator estimator.py --runner subprocess
```

Same score format as Stage 3. If your score drops noticeably, you've found a bug masked by in-process state.

## When you're ready

Move on to [Stage 5: Docker runner](stage-5-run-docker.md) (placeholder for now).
