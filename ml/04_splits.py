"""Divisoes em folds: (a) por cantor (nenhum cantor em treino e teste) e (b) estilo do artigo (aleatoria por arquivo).

Comparar (a) com (b) mostra quanto o resultado do artigo pode estar inflado por vazamento de cantor.
"""
import argparse

import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold

from common import CLASSES, OUT_DIR, SEED, load_metadata


def main(n_splits: int):
    OUT_DIR.mkdir(exist_ok=True)
    df = load_metadata(verbose=True)[["file_name", "singer_id", "name"]].copy()

    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    df["fold_singer"] = -1
    for k, (_, te) in enumerate(sgk.split(df, df["name"], groups=df["singer_id"])):
        df.loc[df.index[te], "fold_singer"] = k

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    df["fold_paper"] = -1
    for k, (_, te) in enumerate(kf.split(df)):
        df.loc[df.index[te], "fold_paper"] = k

    # verificacoes
    for k in range(n_splits):
        te = df[df.fold_singer == k]
        tr = df[df.fold_singer != k]
        assert not set(te.singer_id) & set(tr.singer_id), "vazamento de cantor!"
        faltam = [c for c in CLASSES if c not in set(te.name)]
        print(f"fold {k}: teste={len(te)} arquivos, {te.singer_id.nunique()} cantores, classes ausentes no teste: {faltam}")
    leak = sum(len(set(df[df.fold_paper == k].singer_id) & set(df[df.fold_paper != k].singer_id)) for k in range(n_splits))
    print("estilo artigo: cantores presentes em treino e teste (soma dos folds):", leak)

    df.to_csv(OUT_DIR / "splits.csv", index=False)
    print("salvo:", OUT_DIR / "splits.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-splits", type=int, default=5)
    main(p.parse_args().n_splits)
