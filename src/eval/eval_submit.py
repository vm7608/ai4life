import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


gt_path = "/home/manhckv/manhckv/ai4life/soutput/ground_truth.csv"

dt_path = "/home/manhckv/manhckv/ai4life/soutput/predict_segment.csv"

gt_df = pd.read_csv(gt_path)
dt_df = pd.read_csv(dt_path)

gt = gt_df["predict"].tolist()
dt = dt_df["predict"].tolist()

accuracy = accuracy_score(gt, dt)
f1_macro = f1_score(gt, dt, average="macro")
f1_weighted = f1_score(gt, dt, average="weighted")

print(f"Accuracy: {accuracy}")
print(f"F1 macro: {f1_macro}")
print(f"F1 weighted: {f1_weighted}")
print(classification_report(gt, dt))
