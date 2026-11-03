"""Junta as notas dos avaliadores (avaliacoes.csv) ao manifesto: rating = media por cover.

Entradas:  --ratings avaliacoes.csv (formato longo: codigo, avaliador, nota...  OU largo: codigo, [musica], uma coluna por avaliador)
           --mapa avaliacao_mapa.csv   --manifest manifesto_covers.csv
Saida:     manifesto_com_notas.csv (mesmo manifesto, com a coluna rating preenchida nos covers)
Imprime tambem quantos avaliadores ha e a concordancia media entre eles (Spearman par a par).
Uso:  python tools/aggregate_ratings.py --ratings avaliacoes.csv --mapa avaliacao_mapa.csv --manifest manifesto_covers.csv
"""
import argparse
import itertools
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def main(a):
    r = pd.read_csv(a.ratings, encoding="utf-8-sig")
    r.columns = [str(c).strip() for c in r.columns]
    nota_cols = [c for c in r.columns if c.lower().startswith("nota")]
    if "avaliador" in r.columns and nota_cols:                       # formato longo: codigo, avaliador, nota...
        r = r.rename(columns={nota_cols[0]: "nota"})[["codigo", "avaliador", "nota"]]
    else:                                                            # formato largo: uma coluna por avaliador
        ign = {"codigo", "música", "musica"}
        raters = [c for c in r.columns if c.lower() not in ign]
        r = r.melt(id_vars="codigo", value_vars=raters, var_name="avaliador", value_name="nota")
    r["nota"] = pd.to_numeric(r["nota"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    r = r.dropna(subset=["nota"])
    col = "nota"
    mapa = pd.read_csv(a.mapa)
    r = r.merge(mapa[["codigo", "arquivo_cover"]], on="codigo")
    mean = r.groupby("arquivo_cover")[col].agg(["mean", "count"])
    print("Nota media por cover:\n", mean.round(2).to_string())
    wide = r.pivot_table(index="codigo", columns="avaliador", values=col)
    rhos = [spearmanr(wide[x], wide[y], nan_policy="omit")[0] for x, y in itertools.combinations(wide.columns, 2)
            if wide[[x, y]].dropna().shape[0] >= 4]
    print(f"\nAvaliadores: {wide.shape[1]} | concordancia media entre pares (Spearman): " + (f"{sum(rhos) / len(rhos):.2f}" if rhos else "n/d (precisa de 2+ avaliadores)"))
    m = pd.read_csv(a.manifest)
    m["rating"] = m["recording"].map(lambda p: mean["mean"].get(Path(str(p)).name, float("nan")))
    m.to_csv("manifesto_com_notas.csv", index=False)
    print("\nsalvo: manifesto_com_notas.csv (use: python 13_compare_pitch.py --manifest manifesto_com_notas.csv)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", required=True)
    p.add_argument("--mapa", required=True)
    p.add_argument("--manifest", required=True)
    main(p.parse_args())
