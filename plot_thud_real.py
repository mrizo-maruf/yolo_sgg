import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CSV paths
# -----------------------------
csv_paths = [
    "/home/yehia/rizo/THUD_Robot/Real_Scenes/1008/static/Capture_2/Capture_2_metrics.csv",
    "/home/yehia/rizo/THUD_Robot/Real_Scenes/1008/static/Capture_1/Capture_1_metrics.csv",
    "/home/yehia/rizo/THUD_Robot/Real_Scenes/1004/static/Capture_1/Capture_1_metrics.csv",
    "/home/yehia/rizo/THUD_Robot/Real_Scenes/1004/static/Capture_2/Capture_2_metrics.csv",
    "/home/yehia/rizo/THUD_Robot/Real_Scenes/21L/static/Capture_1/Capture_1_metrics.csv",
    "/home/yehia/rizo/THUD_Robot/Real_Scenes/10L/static/Capture_2/Capture_2_metrics.csv",
    "/home/yehia/rizo/THUD_Robot/Real_Scenes/10L/static/Capture_1/Capture_1_metrics.csv",
]

# -----------------------------
# Metrics to average
# -----------------------------
metrics_to_extract = [
    "T_mIoU",
    "T_SR",
    "MOTA",
    "MOTP",
    "ID_consistency"
]

# -----------------------------
# Read + collect metrics
# -----------------------------
all_metrics = {metric: [] for metric in metrics_to_extract}

for path in csv_paths:
    df = pd.read_csv(path)

    for metric in metrics_to_extract:
        value = df.loc[df["Metric"] == metric, "Value"].values
        if len(value) > 0:
            all_metrics[metric].append(float(value[0]))

# -----------------------------
# Compute averages
# -----------------------------
averages = {}
for metric in metrics_to_extract:
    if len(all_metrics[metric]) > 0:
        averages[metric] = sum(all_metrics[metric]) / len(all_metrics[metric])
    else:
        averages[metric] = 0.0

# -----------------------------
# Visualization
# -----------------------------
metric_names = list(averages.keys())
metric_values = list(averages.values())

plt.figure()
plt.bar(metric_names, metric_values, label="Average Metrics")

plt.title("RealScenes, THUD Dataset")
plt.grid(True)
plt.xlabel("Metrics")
plt.ylabel("Average Value")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -----------------------------
# Print numeric results
# -----------------------------
print("Average Metrics Across All Sequences:")
for k, v in averages.items():
    print(f"{k}: {v:.4f}")