import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "student": [f"S{i}" for i in range(1, 13)],
    "study_hours": [1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0, 7.5, 8.0, 9.0],
    "exam_score": [52, 56, 63, 66, 70, 76, 79, 82, 88, 90, 92, 96]
})

pearson_corr = df["study_hours"].corr(df["exam_score"])
spearman_corr = df["study_hours"].corr(df["exam_score"], method="spearman")

print(f"Pearson Correlation:  {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

# Visualizations: Scatter plot + Trend line
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].scatter(df["study_hours"], df["exam_score"], color="royalblue")
axes[0].set_title("Scatter Plot: Study Hours vs. Exam Score")
axes[0].set_xlabel("Study Hours")
axes[0].set_ylabel("Exam Score")

m, b = np.polyfit(df["study_hours"], df["exam_score"], 1)
axes[1].scatter(df["study_hours"], df["exam_score"], color="teal", label="Students")
axes[1].plot(df["study_hours"], m * df["study_hours"] + b, color="crimson", label="Linear Trend")
axes[1].set_title("Trend Line Overlay")
axes[1].set_xlabel("Study Hours")
axes[1].legend()

plt.tight_layout()
plt.show()
