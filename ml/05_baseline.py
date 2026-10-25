"""Baseline classico: features de log-Mel/MFCC + regressao logistica e random forest, validacao cruzada."""
import argparse

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, OUT_DIR, SEED


def handcrafted(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)                       # (N, mels, frames)
    db = X * 100 - 100
    mfcc = dct(db, type=2, axis=1, norm="ortho")[:, :20, :]
    flux = np.abs(np.diff(X, axis=2)).mean(axis=(1, 2))[:, None]
    return np.hstack([
        X.mean(2), X.std(2),                       # espectro medio e variacao por banda
        mfcc.mean(2), mfcc.std(2),
        np.abs(np.diff(mfcc, axis=2)).mean(2),     # variacao dos MFCC
        flux,
    ])


def models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=SEED),
    }


def main(split: str, task: str):
    X = np.load(OUT_DIR / "X.npy")
    w = pd.read_csv(OUT_DIR / "windows.csv")
    sp = pd.read_csv(OUT_DIR / "splits.csv")[["file_name", f"fold_{split}"]].rename(columns={f"fold_{split}": "fold"})
    w = w.merge(sp, on="file_name", how="left")
    y = w["ybin" if task == "binary" else "y4"].to_numpy()
    F = handcrafted(X)
    folds = sorted(w["fold"].dropna().unique())
    labels = [0, 1] if task == "binary" else [0, 1, 2, 3]

    rows, cms = [], {}
    for name in models():
        cms[name] = np.zeros((len(labels), len(labels)))
    for k in folds:
        tr, te = (w["fold"] != k).to_numpy(), (w["fold"] == k).to_numpy()
        for name, m in models().items():
            m.fit(F[tr], y[tr])
            proba = m.predict_proba(F[te])
            pred = m.classes_[proba.argmax(1)]
            # nivel de arquivo: media das probabilidades das janelas
            pf = pd.DataFrame(proba, columns=m.classes_).assign(file_name=w.loc[te, "file_name"].to_numpy())
            fpred = pf.groupby("file_name").mean().idxmax(axis=1).astype(int)
            ftrue = w.loc[te].groupby("file_name")[("ybin" if task == "binary" else "y4")].first().reindex(fpred.index).astype(int)
            rows.append(dict(
                split=split, task=task, model=name, fold=int(k), n_janelas=int(te.sum()),
                acc=accuracy_score(y[te], pred), bal_acc=balanced_accuracy_score(y[te], pred),
                f1_macro=f1_score(y[te], pred, average="macro"),
                acc_arquivo=accuracy_score(ftrue, fpred), bal_acc_arquivo=balanced_accuracy_score(ftrue, fpred),
            ))
            cms[name] += confusion_matrix(y[te], pred, labels=labels)
            print(f"fold {int(k)} {name}: bal_acc={rows[-1]['bal_acc']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / f"baseline_{task}_{split}.csv", index=False)
    cols = ["acc", "bal_acc", "f1_macro", "acc_arquivo", "bal_acc_arquivo"]
    summ = res.groupby("model")[cols].agg(["mean", "std"]).round(3)
    print(f"\n=== {task} | split={split} (media e desvio entre folds) ===\n", summ.to_string())
    names = ["limpa", "distorcida"] if task == "binary" else CLASSES
    for name, cm in cms.items():
        pd.DataFrame(cm, index=names, columns=names).to_csv(OUT_DIR / f"cm_{task}_{split}_{name}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["singer", "paper"], default="singer")
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    a = p.parse_args()
    main(a.split, a.task)
