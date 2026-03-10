import pickle, json, yaml, librosa, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

N_ESTIMATORS = params["n_estimators"]
N_MFCC       = params["n_mfcc"]
SAMPLE_RATE  = params["sample_rate"]
TEST_SPLIT   = params["test_split"]

Path("models").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

print("Loading data...")
X, y = [], []
for wav in Path("data/positive").glob("*.wav"):
    try:
        audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        X.append(np.mean(mfcc.T, axis=0))
        y.append(1)
    except: pass
for wav in Path("data/negative").glob("*.wav"):
    try:
        audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        X.append(np.mean(mfcc.T, axis=0))
        y.append(0)
    except: pass

X, y = np.array(X), np.array(y)
print(f"Dataset: {len(X)} samples (pos={y.sum()}, neg={(1-y).sum()})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Training...")
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
metrics = {
    "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
    "recall"   : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    "f1_score" : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    "train_samples": int(len(X_train)),
    "test_samples" : int(len(X_test)),
    "positive_samples": int(y.sum()),
    "negative_samples": int((1-y).sum()),
}
print(f"F1: {metrics['f1_score']} | Precision: {metrics['precision']} | Recall: {metrics['recall']}")

pickle.dump(clf, open("models/model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Done!")
