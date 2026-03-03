# Use Circuit Explorer

## 🚀 When to use this page

Use this page when you want visual intuition about circuit behavior and estimator error patterns.

Circuit Explorer is optional and is not the submission interface.

## Do this now

```bash
cd tools/circuit-explorer
npm install
npm run dev
```

Open `http://localhost:5173`.

## ✅ Expected outcome

You can interactively inspect circuit structure, layer behavior, and estimator comparisons.

## Suggested workflow

1. Start with small width/depth.
2. Vary seed to inspect structural changes.
3. Compare estimator behavior across layers.
4. Locate where errors concentrate.
5. Convert observations into Python estimator heuristics.

Official score semantics still come from:

```bash
cestim run --estimator <path> --runner subprocess
```

## 🛠 Common first failure

Symptom: app does not start due to missing Node dependencies.

Fix: run `npm install` in `tools/circuit-explorer` and retry `npm run dev`.

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Problem Setup](../concepts/problem-setup.md)
