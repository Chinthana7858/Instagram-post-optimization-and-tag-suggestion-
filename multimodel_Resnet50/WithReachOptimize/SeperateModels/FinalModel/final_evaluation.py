import pandas as pd

# Load CSV
csv_path = r"E:\FinalData\evaluation_output.csv"
df = pd.read_csv(csv_path)

# Extract columns
prev = df['Previous_IHC_h']
pred = df['Predicted_IHC_h']

# Identify and remove rows where Previous_IHC_h is zero
mask = prev != 0
zero_count = (~mask).sum()

prev_nonzero = prev[mask]
pred_nonzero = pred[mask]

# Calculate metrics
avg_gain_percent = ((pred_nonzero - prev_nonzero) / prev_nonzero * 100).mean()
avg_absolute_improvement = (pred_nonzero - prev_nonzero).mean()
avg_improvement_ratio = (pred_nonzero / prev_nonzero).mean()

# Print results
print("Optimization Evaluation:")
print(f"Average Gain (%): {avg_gain_percent:.2f}%")
print(f"Average Absolute Improvement: {avg_absolute_improvement:.4f}")
print(f"Average Improvement Ratio: {avg_improvement_ratio:.4f}")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": ["Average Gain (%)", "Average Absolute Improvement", "Average Improvement Ratio"],
    "Value": [avg_gain_percent, avg_absolute_improvement, avg_improvement_ratio]
})
output_path = "ihc_optimization_summary.csv"
metrics_df.to_csv(output_path, index=False)
print(f"\n📂 Summary saved to {output_path}")
