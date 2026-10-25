"""CNN pequena sobre log-Mel (64x100) com validacao cruzada por cantor.

Mesmos folds, metricas e formato de saida do baseline (05_baseline.py), para comparar lado a lado.
Uso:
    python 06_cnn.py --split singer --task binary
    python 06_cnn.py --split singer --task binary --no-aug      # ablacao: sem augmentation
    python 06_cnn.py --split singer --task multiclass
    python 06_cnn.py --split singer --task binary --folds 0     # teste rapido (so o fold 0)
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from common import CLASSES, OUT_DIR, SEED


class SmallCNN(nn.Module):
    """4 blocos conv-BN-ReLU-pool + pooling global. ~60 mil parametros: leve para rodar no navegador."""

    def __init__(self, n_classes: int, channels=(16, 32, 64, 64), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Augmentation na escala original do log-Mel (0..~1, 100 unidades = 100 dB). x: (B,1,M,T)."""
    B, _, M, T = x.shape
    x = x + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.2      # ganho aleatorio (+-10 dB)
    x = x + 0.02 * torch.randn_like(x)                                  # ruido leve
    for i in range(B):                                                  # SpecAugment: 1 mascara de frequencia + 1 de tempo
        fw, tw = np.random.randint(0, 9), np.random.randint(0, 16)
        f0, t0 = np.random.randint(0, M - fw + 1), np.random.randint(0, T - tw + 1)
        m = x[i].mean()
        x[i, :, f0:f0 + fw, :] = m
        x[i, :, :, t0:t0 + tw] = m
    return x


@torch.no_grad()
def predict_proba(model, X, mean, std, device, batch=512):
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = ((X[s:s + batch].to(device) - mean) / std)
        out.append(torch.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()


def train_fold(X, y, fit_idx, val_idx, n_classes, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = X[fit_idx].mean().item(), X[fit_idx].std().item()
    counts = np.bincount(y[fit_idx], minlength=n_classes).astype(np.float32)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=device)
    model = SmallCNN(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    yt = torch.from_numpy(y)

    best, best_state, wait = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = fit_idx[np.random.permutation(len(fit_idx))]
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            xb = X[idx].to(device)
            if not args.no_aug:
                xb = augment(xb)
            xb = (xb - mean) / std
            loss = loss_fn(model(xb), yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict_proba(model, X[val_idx], mean, std, device).argmax(1)
        score = balanced_accuracy_score(y[val_idx], pv)
        sched.step(1 - score)
        if score > best:
            best, wait = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, mean, std, best, ep + 1


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = torch.from_numpy(np.load(OUT_DIR / "X.npy").astype(np.float32)).unsqueeze(1)
    w = pd.read_csv(OUT_DIR / "windows.csv")
    sp = pd.read_csv(OUT_DIR / "splits.csv")[["file_name", f"fold_{args.split}"]].rename(columns={f"fold_{args.split}": "fold"})
    w = w.merge(sp, on="file_name", how="left")
    ycol = "ybin" if args.task == "binary" else "y4"
    y = w[ycol].to_numpy().astype(np.int64)
    n_classes = 2 if args.task == "binary" else 4
    labels = list(range(n_classes))
    tag = "cnn" + ("_noaug" if args.no_aug else "")
    (OUT_DIR / "models").mkdir(exist_ok=True)
    print(f"device={device} | janelas={len(X)} | parametros={sum(p.numel() for p in SmallCNN(n_classes).parameters())}")

    folds = sorted(w["fold"].dropna().unique())
    if args.folds:
        folds = [f for f in folds if int(f) in args.folds]
    rows, cm = [], np.zeros((n_classes, n_classes))
    for k in folds:
        t0 = time.time()
        tr = np.where(w["fold"] != k)[0]
        te = np.where(w["fold"] == k)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED + int(k))
        a, b = next(gss.split(tr, groups=w["singer_id"].to_numpy()[tr]))   # validacao = cantores separados do treino
        fit_idx, val_idx = tr[a], tr[b]
        model, mean, std, best_val, n_ep = train_fold(X, y, fit_idx, val_idx, n_classes, args, device, SEED + int(k))
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "task": args.task}, OUT_DIR / "models" / f"{tag}_{args.task}_{args.split}_fold{int(k)}.pt")

        proba = predict_proba(model, X[te], mean, std, device)
        pred = proba.argmax(1)
        pf = pd.DataFrame(proba, columns=labels).assign(file_name=w.loc[te, "file_name"].to_numpy())
        fpred = pf.groupby("file_name").mean().idxmax(axis=1).astype(int)
        ftrue = w.iloc[te].groupby("file_name")[ycol].first().reindex(fpred.index).astype(int)
        rows.append(dict(
            split=args.split, task=args.task, model=tag, fold=int(k), n_janelas=len(te), epocas=n_ep, val_bal_acc=best_val,
            acc=accuracy_score(y[te], pred), bal_acc=balanced_accuracy_score(y[te], pred), f1_macro=f1_score(y[te], pred, average="macro"),
            acc_arquivo=accuracy_score(ftrue, fpred), bal_acc_arquivo=balanced_accuracy_score(ftrue, fpred),
        ))
        cm += confusion_matrix(y[te], pred, labels=labels)
        print(f"fold {int(k)}: {n_ep} epocas, val_bal_acc={best_val:.3f}, teste bal_acc={rows[-1]['bal_acc']:.3f} ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / f"{tag}_{args.task}_{args.split}.csv", index=False)
    cols = ["acc", "bal_acc", "f1_macro", "acc_arquivo", "bal_acc_arquivo"]
    print(f"\n=== {tag} | {args.task} | split={args.split} (media e desvio entre folds) ===\n", res[cols].agg(["mean", "std"]).round(3).to_string())
    names = ["limpa", "distorcida"] if args.task == "binary" else CLASSES
    pd.DataFrame(cm, index=names, columns=names).to_csv(OUT_DIR / f"cm_{args.task}_{args.split}_{tag}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["singer", "paper"], default="singer")
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-aug", action="store_true", help="desliga a augmentation (ablacao)")
    p.add_argument("--folds", type=int, nargs="*", help="rodar so estes folds (teste rapido)")
    p.add_argument("--device", default=None)
    main(p.parse_args())
