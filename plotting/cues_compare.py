import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------
# Data from the tables (Emotional & Environmental cues)
# ----------------------------------------------

models = ["Dense", "CNN-Attn", "CNN-LSTM", "LSTM", "Attn"]

# Test accuracies
emotional_test = [54.37, 39.37, 34.37, 58.12, 65.00]
environment_test = [40.62, 33.12, 35.00, 38.75, 39.37]

# ----------------------------------------------
# Plot Configuration
# ----------------------------------------------

x = np.arange(len(models))
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 5))

# Emotional vs Environmental test accuracy bars
ax.bar(x - width/2, emotional_test, width, label='Emotional Test Accuracy', color='tab:blue')
ax.bar(x + width/2, environment_test, width, label='Environmental Test Accuracy', color='tab:orange')

# ----------------------------------------------
# Graph Styling
# ----------------------------------------------

ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
# ax.set_title("Comparison of Test Accuracy for Emotional vs Environmental Cues", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend()

plt.tight_layout()

# ----------------------------------------------
# Save the figure
# ----------------------------------------------

plt.savefig("./cue_test_accuracy_comparison.png", dpi=300)
plt.show()

print("Bar graph saved as './cue_test_accuracy_comparison.png'")
