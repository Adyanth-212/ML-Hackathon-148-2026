
import pandas as pd
from pathlib import Path

results = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.9582,
        "Macro F1": 0.894184,
        "Execution Time (s)": 2.25
    },
    {
        "Model": "Extra Trees",
        "Accuracy": 0.9418,
        "Macro F1": 0.860096,
        "Execution Time (s)": 2.20
    },
    {
        "Model": "Decision Tree",
        "Accuracy": 0.9526,
        "Macro F1": 0.884180,
        "Execution Time (s)": 2.21
    }
]

df = pd.DataFrame(results)

output = Path("experiments/results/sudhe_model_comparison.csv")
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)

print(df.to_string(index=False))
print("\nComparison saved to:", output)
print("\nNote: Scores are training-sample results, not held-out test results.")
