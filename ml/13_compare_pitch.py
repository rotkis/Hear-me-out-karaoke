"""Compara os metodos de pitch A, B1, B2 e D sobre as MESMAS gravacoes.

Manifesto CSV (colunas):
  recording   caminho da gravacao do cantor (wav/flac/mp3)
  quem        rotulo livre (eu, amigo, original...)               [opcional]
  tonic       tonica da musica (C, C#, Db, D, ... B)               [para B1/B2]
  mode        major ou minor                                        [para B1/B2]
  reference   caminho da voz de referencia separada (vocals.wav)   [opcional, para D]
  rating      nota humana (qualquer escala numerica)                [opcional]
Uso:  python 13_compare_pitch.py --manifest meu_manifesto.csv
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pitch
import scoring
from common import OUT_DIR, SR


def main(a):
    OUT_DIR.mkdir(exist_ok=True)
    m = pd.read_csv(a.manifest)
    def limpa(p):                                 # o `ls` mostra aspas em nomes com espaco; elas nao fazem parte do nome
        if not isinstance(p, str):
            return p
        p = p.strip()
        return p if Path(p).expanduser().exists() else p.replace("'", "").replace('"', "")

    for c in ("recording", "reference"):
        if c in m:
            m[c] = m[c].map(limpa)
    missing = [(c, p) for c in ("recording", "reference") if c in m for p in m[c].dropna().astype(str)
               if p and not Path(p).expanduser().exists()]
    if missing:
        print("Arquivos nao encontrados (ajuste os caminhos no manifesto):")
        for c, p in missing:
            print(f"  [{c}] {p}")
        print("Dica: rode `ls dados/covers` e copie os nomes exatos (com espacos e extensoes).")
        raise SystemExit(1)
    ref_cache, rows = {}, []
    for r in m.to_dict("records"):
        y, _ = librosa.load(r["recording"], sr=SR, mono=True)
        tr = pitch.track(y)
        w = scoring.weights(tr)
        tonic, mode = scoring.NOTE[str(r["tonic"])], str(r["mode"])
        row = dict(recording=r["recording"], quem=r.get("quem", ""), frames_confiaveis=int((w >= 0.5).sum()),
                   A_grade=scoring.score_grid(tr),
                   B1_tom_bin=scoring.score_key(tr, tonic, mode),
                   B2_tom_cont=scoring.score_key(tr, tonic, mode, continuous=True))
        ref = r.get("reference")
        if isinstance(ref, str) and ref:
            if ref not in ref_cache:
                yr, _ = librosa.load(ref, sr=SR, mono=True)
                ref_cache[ref] = pitch.track(yr)
            row["D_referencia"] = scoring.score_ref(tr, ref_cache[ref], band=a.band)
            d2 = scoring.ref_lag_detail(tr, ref_cache[ref])
            row["D2_lag_global"], row["D2_atraso_s"], row["D2_cobertura"] = d2["score"], d2["atraso_s"], d2["cobertura"]
            d2b = scoring.ref_env_detail(tr, ref_cache[ref])
            row["D2b_env_lag"], row["D2b_atraso_s"], row["D2b_corr_env"], row["D2b_cobertura"] = d2b["score"], d2b["atraso_s"], d2b["corr_env"], d2b["cobertura"]
        if "rating" in r and not pd.isna(r["rating"]):
            row["rating"] = r["rating"]
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "pitch_compare.csv", index=False)
    pd.set_option("display.width", 200)
    res.insert(0, "arquivo", res["recording"].map(lambda p: (Path(p).parent.name + "/" if Path(p).stem == "vocals" else "") + Path(p).stem[:26]))
    show = res.drop(columns=["recording"] + (["quem"] if res["quem"].eq("").all() else [])).round(3)
    print(show.to_string(index=False))

    methods = [c for c in ("A_grade", "B1_tom_bin", "B2_tom_cont", "D_referencia", "D2_lag_global", "D2b_env_lag") if c in res]
    if len(res) >= 5:
        print("\nConcordancia entre metodos (Spearman):")
        print(res[methods].corr(method="spearman").round(2).to_string())
    if "rating" in res and res["rating"].notna().sum() >= 5:
        print("\nCorrelacao com a nota humana (Spearman):")
        for c in methods:
            ok = res[[c, "rating"]].dropna()
            rho = spearmanr(ok[c], ok["rating"])[0] if len(ok) >= 5 else float("nan")
            print(f"  {c}: {rho:.2f}  (n={len(ok)})")
    print("\nsalvo em", OUT_DIR / "pitch_compare.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--band", type=float, default=0.1, help="largura da faixa do DTW no metodo D")
    main(p.parse_args())
