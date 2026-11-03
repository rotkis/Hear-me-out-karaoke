"""Experimento: em que fracao do tempo o pitch e CONFIAVEL, por tecnica vocal, fala e condicoes extras (ex.: rap)?

Mede, quadro a quadro, a confianca do YIN (minimo do CMNDF: quanto MENOR, mais periodico) e, opcionalmente, o pYIN do librosa.
O resultado decide o limiar de confianca da politica "pitch ponderado pela confianca" (SDD, secoes 3.3 e 11.1).

Uso:
    python 10_pitch_confidence.py                                   # so as 4 classes do EMVD
    python 10_pitch_confidence.py --speech-dir ~/dados/mls/dev/audio # + fala
    python 10_pitch_confidence.py --extra rap=~/gravacoes/rap        # + qualquer condicao: NOME=PASTA (repita a opcao)
    python 10_pitch_confidence.py --pyin                             # + pYIN (mais lento)

Saidas em outputs/: pitch_conf_files.csv, pitch_conf_summary.csv, pitch_conf_by_range.csv, pitch_conf.png
"""
import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import AUDIO_DIR, CLASSES, OUT_DIR, SEED, SR, load_metadata

EXTS = {".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a"}
FRAME, HOP = 1024, 160          # 64 ms de janela, passo de 10 ms (mesmo passo do log-Mel)
FMIN, FMAX = 65.0, 1000.0       # ~C2 a ~B5
ACTIVE_RMS = 0.02               # quadros mais fracos que isso (com o arquivo normalizado pelo pico) nao contam
THRESHOLDS = (0.10, 0.20, 0.30)


def yin_cmndf_min(y: np.ndarray):
    """Por quadro: (minimo do CMNDF, f0 em Hz, RMS). Implementacao vetorizada do YIN (de Cheveigne e Kawahara, 2002)."""
    tau_min, tau_max = int(SR / FMAX), int(SR / FMIN)
    W = FRAME // 2
    n = 1 + (len(y) - FRAME) // HOP
    if n <= 0:
        return np.empty(0), np.empty(0), np.empty(0)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    F = y[idx].astype(np.float64)
    rms = np.sqrt((F ** 2).mean(axis=1))

    nfft = 2048
    A = np.fft.rfft(F[:, :W], nfft, axis=1)
    B = np.fft.rfft(F, nfft, axis=1)
    c = np.fft.irfft(np.conj(A) * B, nfft, axis=1)[:, : tau_max + 1]          # sum_j a_j * b_{j+tau}
    e_a = (F[:, :W] ** 2).sum(axis=1, keepdims=True)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(F ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    e_b = cs[:, taus + W] - cs[:, taus]                                         # energia de b em [tau, tau+W)
    d = np.maximum(e_a + e_b - 2 * c, 0)
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1)
    cmndf = np.ones_like(d)
    cmndf[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-12)
    seg = cmndf[:, tau_min: tau_max + 1]
    # f0: primeiro atraso abaixo do limiar absoluto (evita subharmonicos), refinado no minimo local logo depois;
    # sem nenhum abaixo do limiar, usa o minimo global. A CONFIANCA continua sendo o minimo global do CMNDF.
    below = seg < 0.15
    first = np.where(below.any(axis=1), below.argmax(axis=1), seg.argmin(axis=1))
    win = np.minimum(first[:, None] + np.arange(12)[None, :], seg.shape[1] - 1)
    k = win[np.arange(n), np.take_along_axis(seg, win, axis=1).argmin(axis=1)]
    return seg.min(axis=1), SR / (tau_min + k), rms


def file_metrics(y: np.ndarray, use_pyin: bool):
    y = y / (np.abs(y).max() + 1e-9)
    cm, f0, rms = yin_cmndf_min(y)
    act = rms > ACTIVE_RMS
    if act.sum() < 20:
        return None
    cm_a, f0_a = cm[act], f0[act]
    out = {"frames_ativos": int(act.sum()), "cmndf_mediana": float(np.median(cm_a))}
    for t in THRESHOLDS:
        out[f"frac_confiavel_{t:.2f}"] = float((cm_a < t).mean())
    rel = (cm < 0.20) & act                                                      # estabilidade: saltos de oitava entre quadros confiaveis vizinhos
    both = rel[1:] & rel[:-1]
    jump = np.abs(np.log2(f0[1:] / f0[:-1])) > 0.5
    out["salto_oitava"] = float(jump[both].mean()) if both.sum() > 10 else np.nan
    if use_pyin:
        _, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=FRAME, hop_length=HOP)
        out["pyin_voiced"] = float(np.mean(voiced[: len(act)][act[: len(voiced)]])) if len(voiced) else np.nan
    return out


def list_audio(folder: str, n_files: int, rng):
    files = [p for p in Path(folder).expanduser().rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"nenhum audio em {folder}")
    pick = rng.permutation(len(files))[:n_files]
    return [files[i] for i in pick]


def main(a):
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    jobs = []                                                       # (condicao, caminho, faixa)
    df = load_metadata(verbose=True)
    for r in df.itertuples(index=False):
        jobs.append((r.name, AUDIO_DIR / r.file_name, getattr(r, "range", "-")))
    if a.speech_dir:
        jobs += [("fala", p, "-") for p in list_audio(a.speech_dir, a.speech_files, rng)]
    for spec in a.extra or []:
        name, folder = spec.split("=", 1)
        jobs += [(name, p, "-") for p in list_audio(folder, a.extra_files, rng)]

    rows = []
    for cond, path, rng_ in tqdm(jobs):
        y, _ = librosa.load(path, sr=SR, mono=True, duration=a.max_sec)
        m = file_metrics(y, a.pyin)
        if m:
            rows.append(dict(condicao=cond, arquivo=Path(path).name, faixa=rng_, **m))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "pitch_conf_files.csv", index=False)

    order = [c for c in CLASSES if c in set(res["condicao"])] + [c for c in res["condicao"].unique() if c not in CLASSES]
    cols = [c for c in res.columns if c.startswith("frac_confiavel") or c in ("cmndf_mediana", "salto_oitava", "pyin_voiced")]
    summ = res.groupby("condicao")[cols].agg(["median", "mean"]).reindex(order)
    summ.insert(0, "arquivos", res.groupby("condicao").size().reindex(order))
    summ.to_csv(OUT_DIR / "pitch_conf_summary.csv")
    print("\n=== Fracao dos quadros ativos com pitch confiavel (CMNDF < limiar) e outras medidas, por condicao ===")
    print(res.groupby("condicao")[cols].median().reindex(order).round(3).to_string())
    print("\n(mediana entre arquivos; arquivos por condicao:", res.groupby("condicao").size().reindex(order).to_dict(), ")")

    emvd = res[res["condicao"].isin(CLASSES)]
    if emvd["faixa"].nunique() > 1:
        by = emvd.groupby(["condicao", "faixa"])["frac_confiavel_0.20"].agg(["median", "count"]).round(3)
        by.to_csv(OUT_DIR / "pitch_conf_by_range.csv")
        print("\n=== EMVD por registro (mediana da fracao confiavel com limiar 0,20) ===\n", by.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([res.loc[res.condicao == c, "frac_confiavel_0.20"].dropna() for c in order], tick_labels=order)
    ax.set_ylabel("fração de quadros com pitch confiável (CMNDF < 0,20)")
    ax.set_title("Pitch confiável por condição (cada ponto = um arquivo)")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pitch_conf.png", dpi=130)
    print("\nsalvo em", OUT_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speech-dir")
    p.add_argument("--speech-files", type=int, default=150)
    p.add_argument("--extra", action="append", metavar="NOME=PASTA", help="condicao extra (ex.: rap=~/gravacoes/rap); repita a opcao")
    p.add_argument("--extra-files", type=int, default=100)
    p.add_argument("--max-sec", type=float, default=20.0, help="usa so os primeiros N segundos de cada arquivo")
    p.add_argument("--pyin", action="store_true", help="tambem mede o pYIN (lento)")
    main(p.parse_args())
