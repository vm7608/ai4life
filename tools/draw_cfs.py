import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ground_truth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
predictions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


draw_cfs = confusion_matrix(ground_truth, predictions)

plt.figure(figsize=(20, 20))
sns.heatmap(draw_cfs, annot=True, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Truth")

# save the plot
plt.savefig("cfs.png")
