"""Teste de sensibilidade dos metodos de pitch: desloca cada cover em +-50/+-100/+-200 cents e mede como cada nota reage.

Ideia (a mesma do artigo de tonalidade de 2025, que deslocou trechos em 100-200 cents): quanto mais deslocado, PIOR deveria ser
a afinacao. Um bom metodo deve cair de forma consistente com o tamanho do deslocamento, em gravacoes reais.

O que se espera (pela checagem sintetica; aqui se confere com canto real):
  A  (grade)      cai em +-50, mas NAO enxerga +-100/+-200 (a grade se repete a cada semitom)
  B1 (tom bin.)   cai em +-100/+-200 (notas saem da escala), pouco em +-50
  B2 (tom cont.)  cai em todos, de forma mais gradual
  D  (referencia) deveria cair em todos, mas o DTW permite casar a nota com outra de mesma classe (nao caiu nos covers reais)
  D2 (ref., atraso global, sem DTW) deve cair em todos: preserva a ordem da melodia
  D2b(ref., atraso pelo envelope de energia) idem, sem poder usar o atraso para compensar o pitch

Usa o mesmo manifesto do 13_compare_pitch.py (so as linhas com quem != original). O D so entra onde houver `reference`.
Uso:  python 14_shift_test.py --manifest manifesto_covers.csv
Saidas em outputs/: shift_test.csv, shift_summary.csv, shift_test.png
"""
import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pitch
import scoring
from common import OUT_DIR, SR

METHODS = ["A_grade", "B1_tom_bin", "B2_tom_cont", "D_referencia", "D2_lag_global", "D2b_env_lag"]


def limpa(p):                                   # o `ls` mostra aspas em nomes com espaco; elas nao fazem parte do nome
    if not isinstance(p, str):
        return p
    p = p.strip()
    return p if Path(p).expanduser().exists() else p.replace("'", "").replace('"', "")


def main(a):
    OUT_DIR.mkdir(exist_ok=True)
    m = pd.read_csv(a.manifest)
    for c in ("recording", "reference"):
        if c in m:
            m[c] = m[c].map(limpa)
    if "quem" in m:
        m = m[m["quem"].astype(str) != "original"]
    shifts = sorted(set(a.shifts) | {0})
    rows, ref_cache = [], {}

    for r in m.to_dict("records"):
        y, _ = librosa.load(r["recording"], sr=SR, mono=True)
        base = pitch.track(y)
        w0 = scoring.weights(base)
        tonic, mode = scoring.NOTE[str(r["tonic"])], str(r["mode"])
        ref = r.get("reference")
        tr_ref = None
        if isinstance(ref, str) and ref:
            if ref not in ref_cache:
                ref_cache[ref] = pitch.track(librosa.load(ref, sr=SR, mono=True)[0])
            tr_ref = ref_cache[ref]
        name = Path(r["recording"]).stem[:26]
        for s in shifts:
            ys = y if s == 0 else librosa.effects.pitch_shift(y, sr=SR, n_steps=s / 100.0, bins_per_octave=12)
            tr = base if s == 0 else pitch.track(ys)
            n = min(len(tr["f0"]), len(base["f0"]))
            both = (scoring.weights(tr)[:n] >= 0.5) & (w0[:n] >= 0.5)
            medido = float(np.median(1200 * np.log2(tr["f0"][:n][both] / base["f0"][:n][both]))) if both.sum() > 20 else float("nan")
            d2 = scoring.ref_lag_detail(tr, tr_ref) if tr_ref is not None else dict(score=float("nan"), atraso_s=float("nan"), cobertura=float("nan"))
            d2b = scoring.ref_env_detail(tr, tr_ref) if tr_ref is not None else dict(score=float("nan"), atraso_s=float("nan"), corr_env=float("nan"), cobertura=float("nan"))
            rows.append(dict(arquivo=name, quem=r.get("quem", ""), deslocamento=s, medido=medido,
                             D2b_env_lag=d2b["score"], D2b_atraso_s=d2b["atraso_s"], D2b_corr_env=d2b["corr_env"], D2b_cobertura=d2b["cobertura"],
                             D2_lag_global=d2["score"], D2_atraso_s=d2["atraso_s"], D2_cobertura=d2["cobertura"],
                             A_grade=scoring.score_grid(tr),
                             B1_tom_bin=scoring.score_key(tr, tonic, mode),
                             B2_tom_cont=scoring.score_key(tr, tonic, mode, continuous=True),
                             D_referencia=scoring.score_ref(tr, tr_ref, band=a.band) if tr_ref is not None else float("nan")))
        print(f"ok: {name}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "shift_test.csv", index=False)
    ms = [c for c in METHODS if res[c].notna().any()]
    pd.set_option("display.width", 200)

    err = (res["medido"] - res["deslocamento"]).abs()
    print(f"\nVerificacao do deslocamento: erro mediano entre o pedido e o medido no proprio audio = {err[res.deslocamento != 0].median():.0f} cents")

    if "D2_lag_global" in ms:
        diag = res.groupby("deslocamento").agg(atraso_mediano_s=("D2_atraso_s", "median"), cobertura_media=("D2_cobertura", "mean"),
                                               cobertura_min=("D2_cobertura", "min"), sem_nota=("D2_lag_global", lambda x: int(x.isna().sum())))
        print("\n=== Diagnostico do D2 por deslocamento (atraso escolhido, cobertura, covers sem nota por cobertura baixa) ===\n", diag.round(3).to_string())

    if "D2b_env_lag" in ms:
        diag = res.groupby("deslocamento").agg(atraso_mediano_s=("D2b_atraso_s", "median"), atraso_min_s=("D2b_atraso_s", "min"), atraso_max_s=("D2b_atraso_s", "max"),
                                               corr_env_media=("D2b_corr_env", "mean"), cobertura_media=("D2b_cobertura", "mean"), sem_nota=("D2b_env_lag", lambda x: int(x.isna().sum())))
        print("\n=== Diagnostico do D2b por deslocamento (o atraso NAO deveria mudar com o deslocamento; corr_env = confianca do alinhamento) ===\n", diag.round(3).to_string())
        z = res[res.deslocamento == 0].set_index("arquivo")[["D2b_atraso_s", "D2b_corr_env", "D2b_cobertura"]]
        print("\nD2b sem deslocamento, por cover:\n", z.round(2).to_string())

    mean = res.groupby("deslocamento")[ms].mean()
    print("\n=== Nota media dos covers por deslocamento (cents) ===\n", mean.round(3).to_string())

    # queda media em relacao ao cover original (deslocamento 0), agrupando +s e -s
    base0 = res[res.deslocamento == 0].set_index("arquivo")[ms]
    rows2 = []
    for s in [x for x in shifts if x > 0]:
        for sign in (-1, 1):
            sub = res[res.deslocamento == sign * s].set_index("arquivo")[ms]
            if sub.empty:
                continue
            d = sub - base0.loc[sub.index]
            rows2.append(dict(deslocamento=sign * s, **{f"{c}": f"{d[c].mean():+.3f} ({(d[c] < 0).sum()}/{d[c].notna().sum()} caem)" for c in ms}))
    print("\n=== Variacao media da nota vs. sem deslocamento (e em quantos covers a nota CAI) ===\n", pd.DataFrame(rows2).to_string(index=False))

    summ = []
    for c in ms:
        rho = []
        for _, g in res.groupby("arquivo"):
            ok = g[["deslocamento", c]].dropna()
            if len(ok) >= 4:
                rho.append(spearmanr(ok["deslocamento"].abs(), ok[c])[0])
        summ.append(dict(metodo=c, rho_medio_abs_deslocamento_vs_nota=float(np.nanmean(rho)) if rho else float("nan")))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT_DIR / "shift_summary.csv", index=False)
    print("\n=== Spearman medio entre |deslocamento| e a nota (esperado: negativo e perto de -1) ===\n", summ.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, ms):
        for _, g in res.groupby("arquivo"):
            ax.plot(g["deslocamento"], g[c], color="#bbb", lw=0.8)
        ax.plot(mean.index, mean[c], color="#3b5bdb", marker="o", lw=2)
        ax.set_title(c)
        ax.set_xlabel("deslocamento (cents)")
    axes[0].set_ylabel("nota (cinza: cada cover; azul: media)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shift_test.png", dpi=130)
    print("\nsalvo em", OUT_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--shifts", type=int, nargs="+", default=[-200, -100, -50, 50, 100, 200])
    p.add_argument("--band", type=float, default=0.1, help="largura da faixa do DTW (metodo D)")
    main(p.parse_args())
