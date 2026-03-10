import pickle, json, yaml, librosa, numpy as np, csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

N_MFCC = params["n_mfcc"]; SAMPLE_RATE = params["sample_rate"]
TEST_SPLIT = params["test_split"]; THRESHOLD = params["threshold"]
Path("metrics").mkdir(exist_ok=True)

clf = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

X, y = [], []
for wav in Path("data/positive").glob("*.wav"):
    try:
        audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
        X.append(np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T, axis=0))
        y.append(1)
    except: pass
for wav in Path("data/negative").glob("*.wav"):
    try:
        audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
        X.append(np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T, axis=0))
        y.append(0)
    except: pass

X, y = np.array(X), np.array(y)
_, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)
X_test_s = scaler.transform(X_test)
y_prob = clf.predict_proba(X_test_s)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

metrics = {
    "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
    "threshold": THRESHOLD,
    "test_samples": int(len(X_test)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}
print(f"F1: {metrics['f1_score']} | AUC: {metrics['auc_roc']}")
with open("metrics/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
with open("metrics/roc_curve.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["fpr","tpr","threshold"])
    for fp,tp,th in zip(fpr,tpr,thresholds): w.writerow([round(fp,4),round(tp,4),round(float(th),4)])
print("Metrics saved!")
