"""Teste de robustez do classificador limpa vs. distorcida: quanto ele piora com audio "sujo"?

Para cada fold (cantores nunca vistos pelo modelo daquele fold), as janelas de teste sao degradadas e reavaliadas:
  limpo            referencia (sem degradacao)
  ruido_rosa_<N>dB ruido rosa com relacao sinal/ruido (SNR) de N dB
  <musica>_<N>dB   uma musica instrumental misturada por baixo da voz, com SNR de N dB (0 dB = mesmo volume que a voz)
  mic_ruim         filtro passa-banda 200-4000 Hz + saturacao leve (simula microfone de notebook/telefone)

Modelos comparados: CNN com augmentation (cnn), CNN sem augmentation (cnn_noaug) e random forest (rf).
Os checkpoints das CNNs vem de 06_cnn.py (outputs/models/); o random forest e retreinado em cada fold.

Pre-requisitos: 03_preprocess.py, 04_splits.py e 06_cnn.py (com e sem --no-aug, task binary, split singer).
Uso:
    python 11_robustness.py --music outputs/stems/htdemucs/Hugs/no_vocals.wav --music outputs/stems/htdemucs/Rosa/no_vocals.wav --music outputs/stems/htdemucs/LP/no_vocals.wav
    (o nome da condicao vem da pasta: musica_Hugs_10dB etc.; ou use NOME=CAMINHO, ex.: --music metal=outputs/stems/htdemucs/Hugs/no_vocals.wav)
    python 11_robustness.py --music ~/musicas/x.wav --max-windows 800        # teste rapido
    python 11_robustness.py                                                  # sem musica: so ruido e microfone
Use musicas que voce possa usar (licenca livre). Saidas em outputs/: robustness_results.csv, robustness_summary.csv, robustness.png
"""
import argparse
import importlib.util
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.metrics import balanced_accuracy_score, recall_score
from tqdm import tqdm

from common import AUDIO_DIR, N_FRAMES, N_MELS, OUT_DIR, ROOT, SEED, SR, TARGET_RMS, load_metadata


def _load(file: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pre = _load("03_preprocess.py", "pre03")
cnn = _load("06_cnn.py", "cnn06")
base = _load("05_baseline.py", "base05")


# ---------------------------------------------------------------- degradacoes (operam em janelas de 1 s)
def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) + 1e-12


def pink_noise(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1
    y = np.fft.irfft(spec / np.sqrt(f), n)
    return y / (y.std() + 1e-12)


def mix_at_snr(voice, other, snr_db):
    return voice + other * (rms(voice) / (rms(other) * 10 ** (snr_db / 20)))


SOS_MIC = butter(4, [200, 4000], btype="band", fs=SR, output="sos")


def degrade(w, cond, rng):
    kind = cond[0]
    if kind == "noise":
        out = mix_at_snr(w, pink_noise(len(w), rng), cond[1])
    elif kind == "music":
        m = cond[1]
        for _ in range(5):                                   # evita trechos silenciosos da musica
            s = int(rng.integers(0, max(1, len(m) - len(w))))
            seg = m[s:s + len(w)]
            if len(seg) == len(w) and rms(seg) > 1e-4:
                break
        out = mix_at_snr(w, seg, cond[2])
    elif kind == "mic":
        out = np.tanh(3.0 * sosfilt(SOS_MIC, w)) / 3.0
    else:
        out = w
    return (out * (TARGET_RMS / rms(out))).astype(np.float32)   # renormaliza por janela, como no uso real


def build_conditions(snrs, musics):
    conds = [("limpo", ("clean",))]
    conds += [(f"ruido_rosa_{s}dB", ("noise", s)) for s in snrs]
    for name, m in musics.items():
        conds += [(f"{name}_{s}dB", ("music", m, s)) for s in snrs]
    conds.append(("mic_ruim", ("mic",)))
    return conds


def nome_musica(spec, usados):
    """--music CAMINHO ou NOME=CAMINHO. Para arquivos de nome generico (no_vocals, vocals...), usa o nome da pasta."""
    if "=" in spec:
        nome, caminho = spec.split("=", 1)
    else:
        caminho = spec
        p = Path(caminho).expanduser()
        nome = p.parent.name if p.stem.lower() in {"no_vocals", "vocals", "accompaniment"} else p.stem
    base_nome = nome = "musica_" + nome.strip()
    i = 2
    while nome in usados:                                   # evita que duas musicas com o mesmo nome se sobrescrevam
        nome, i = f"{base_nome}_{i}", i + 1
    return nome, caminho


# ---------------------------------------------------------------- principal
def main(a):
    rng_global = np.random.default_rng(SEED)
    musics = {}
    for spec in a.music or []:
        nome, caminho = nome_musica(spec, musics)
        y, _ = librosa.load(Path(caminho).expanduser(), sr=SR, mono=True)
        musics[nome] = y
        print(f"musica carregada: {nome} ({caminho}, {len(y) / SR:.0f} s)")
    if not musics:
        print("AVISO: nenhuma --music informada; testando so ruido e microfone.")
    conds = build_conditions(a.snr, musics)

    meta = load_metadata()
    ybin = dict(zip(meta["file_name"], meta["ybin"]))
    sp = pd.read_csv(OUT_DIR / "splits.csv")
    wtrain = pd.read_csv(OUT_DIR / "windows.csv").merge(sp[["file_name", "fold_singer"]], on="file_name")
    X_all = np.load(OUT_DIR / "X.npy")
    folds = sorted(sp["fold_singer"].unique())
    if a.folds:
        folds = [f for f in folds if int(f) in a.folds]
    device = None
    rows = []

    for k in folds:
        k = int(k)
        rng = np.random.default_rng(SEED + k)
        models = {}                                                    # nome -> funcao(X[N,M,T]) -> predicoes
        for tag in ("cnn", "cnn_noaug"):
            ck = OUT_DIR / "models" / f"{tag}_binary_singer_fold{k}.pt"
            if tag in a.models and ck.exists():
                import torch
                device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                d = torch.load(ck, map_location=device)
                net = cnn.SmallCNN(2).to(device)
                net.load_state_dict(d["state_dict"])
                models[tag] = (lambda X, net=net, d=d: cnn.predict_proba(
                    net, torch.from_numpy(X.astype(np.float32)).unsqueeze(1), d["mean"], d["std"], device).argmax(1))
            elif tag in a.models:
                print(f"fold {k}: checkpoint {ck.name} nao encontrado; pulando {tag}")
        if "rf" in a.models:
            tr = (wtrain["fold_singer"] != k).to_numpy()
            rf = base.models()["rf"]
            rf.fit(base.handcrafted(X_all[tr]), wtrain.loc[tr, "ybin"].to_numpy())
            models["rf"] = lambda X, rf=rf: rf.predict(base.handcrafted(X))
        if not models:
            raise SystemExit("nenhum modelo disponivel (rode 06_cnn.py antes ou use --models rf)")

        # janelas de teste deste fold (audio -> janelas, igual ao 03_preprocess.py)
        wins, ys = [], []
        for fn in sp.loc[sp["fold_singer"] == k, "file_name"]:
            y, _ = librosa.load(AUDIO_DIR / fn, sr=SR, mono=True)
            for _, w in pre.windows(y):
                wins.append(w.astype(np.float32))
                ys.append(ybin[fn])
        ys = np.array(ys)
        if a.max_windows and len(wins) > a.max_windows:
            keep = rng.choice(len(wins), a.max_windows, replace=False)
            wins, ys = [wins[i] for i in keep], ys[keep]
        print(f"fold {k}: {len(wins)} janelas de teste, modelos: {list(models)}")

        for cname, cond in tqdm(conds, desc=f"fold {k}"):
            Xc = np.stack([pre.logmel(degrade(w, cond, rng)) for w in wins]).astype(np.float32)
            for mname, predict in models.items():
                pred = predict(Xc)
                rows.append(dict(modelo=mname, condicao=cname, fold=k, n=len(ys),
                                 bal_acc=balanced_accuracy_score(ys, pred),
                                 recall_limpa=recall_score(ys, pred, pos_label=0, zero_division=0),
                                 recall_distorcida=recall_score(ys, pred, pos_label=1, zero_division=0)))

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "robustness_results.csv", index=False)
    order = [c for c, _ in conds]
    summ = res.groupby(["condicao", "modelo"])[["bal_acc", "recall_limpa", "recall_distorcida"]].agg(["mean", "std"])
    summ.to_csv(OUT_DIR / "robustness_summary.csv")

    pd.set_option("display.width", 200)
    for col, titulo in [("bal_acc", "ACURACIA BALANCEADA"), ("recall_limpa", "RECALL da voz LIMPA (limpa classificada como limpa)"),
                        ("recall_distorcida", "RECALL da voz DISTORCIDA")]:
        t = res.pivot_table(index="condicao", columns="modelo", values=col, aggfunc="mean").reindex(order)
        print(f"\n=== {titulo} (media entre folds) ===\n", t.round(3).to_string())
    ref = res[res["condicao"] == "limpo"].groupby("modelo")["bal_acc"].mean()
    worst = res.groupby(["condicao", "modelo"])["bal_acc"].mean().unstack()
    print("\nQueda maxima em pontos de acuracia balanceada vs. 'limpo':")
    for m in worst.columns:
        c = worst[m].idxmin()
        print(f"  {m}: {100 * (ref[m] - worst.loc[c, m]):.1f} pontos em '{c}'")

    # grafico: acuracia balanceada x SNR por familia de degradacao
    fams = ["ruido_rosa"] + list(musics)
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(4.2 * (len(fams) + 1), 3.8), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for m in sorted(res["modelo"].unique()):
            ys_ = [res[(res.condicao == f"{fam}_{s}dB") & (res.modelo == m)]["bal_acc"].mean() for s in a.snr]
            ax.plot(a.snr, ys_, marker="o", label=m)
        ax.set_title(fam)
        ax.set_xlabel("SNR (dB)")
        ax.invert_xaxis()
    axes[0][0].set_ylabel("acurácia balanceada")
    axl = axes[0][-1]
    mods = sorted(res["modelo"].unique())
    for i, m in enumerate(mods):
        axl.bar([i - 0.2, i + 0.2], [res[(res.condicao == c) & (res.modelo == m)]["bal_acc"].mean() for c in ("limpo", "mic_ruim")],
                width=0.4, color=["#3b5bdb", "#e8590c"])
    axl.set_xticks(range(len(mods)))
    axl.set_xticklabels(mods)
    axl.set_title("limpo (azul) vs mic_ruim (laranja)")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "robustness.png", dpi=130)
    print("\nsalvo em", OUT_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--music", action="append", help="arquivo de musica instrumental (repita a opcao para varias)")
    p.add_argument("--snr", type=int, nargs="+", default=[20, 10, 0])
    p.add_argument("--models", nargs="+", default=["cnn", "cnn_noaug", "rf"], choices=["cnn", "cnn_noaug", "rf"])
    p.add_argument("--folds", type=int, nargs="*")
    p.add_argument("--max-windows", type=int, default=None, help="subamostra de janelas por fold (teste rapido)")
    main(p.parse_args())
