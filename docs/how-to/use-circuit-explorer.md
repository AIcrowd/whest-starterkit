# Use Circuit Explorer

## When To Use This Page

Use this page to build intuition about circuit behavior and estimator errors.

Circuit Explorer is optional and is not the submission interface.

## Start Explorer

```bash
cd tools/circuit-explorer
npm install
npm run dev
```

Open `http://localhost:5173`.

## Suggested Workflow

1. start with small width/depth,
2. vary seed to see structural changes,
3. compare estimator behavior across layers,
4. inspect where errors concentrate,
5. use observations to improve Python estimator heuristics.

Official score semantics still come from `cestim run`.

## Next

- [Validate, Run, and Package](./validate-run-package.md)
- [Problem Setup](../concepts/problem-setup.md)
