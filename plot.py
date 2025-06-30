import matplotlib.pyplot as plt
import pandas as pd

# Load CSV data
csv_file = "fingerprint_matching_results.csv"  # Change this to your actual CSV file path
df = pd.read_csv(csv_file)

# Unique methods
methods = df["Method"].unique()

# Assign different colors for each method
colors = {
    "Hough Transform": "blue",
    "Genetic Algorithm": "red",
    "Core Point Matching": "green"
}

# Plot each metric separately
metrics = ["Score", "Accuracy", "Precision", "Recall", "F1-Score"]

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)  # Creating subplots in a 2x3 grid
    for method in methods:
        subset = df[df["Method"] == method]
        plt.plot(range(len(subset)), subset[metric], marker='o', linestyle='-', 
                 color=colors[method], label=method)
    
    plt.xlabel("Matching Attempts")
    plt.ylabel(metric)
    plt.title(f"{metric} Over Time")
    plt.legend()
    plt.grid(True)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
