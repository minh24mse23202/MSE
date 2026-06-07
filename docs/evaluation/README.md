# Evaluation

Evaluation tracks:

- routing accuracy;
- context relevance;
- faithfulness proxy or RAGAS faithfulness when configured;
- answer overlap for benchmark answers;
- latency and runtime/cost proxy;
- ablations for `top_k`, hybrid weights, routing thresholds, and prompts.

Run the baseline sample:

```powershell
python scripts/evaluate_sample.py
```

