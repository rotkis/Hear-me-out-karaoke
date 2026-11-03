"""Checagem de implementacao dos metodos A, B1, B2 e D com melodias SINTETICAS (nao e validacao com cantores reais).

Gera uma melodia em Do maior e variantes com erros conhecidos, escreve os wavs + um manifesto e roda 13_compare_pitch.py.
Padrao esperado (se a implementacao estiver certa):
  detune_30   A cai (cantor 30 cents acima da grade); B1 nao ve (continua na escala); B2 cai um pouco
  shift_100   A nao ve (continua na grade); B1/B2/D caem forte (notas fora da escala / classe errada)
  wrong_scale B1/B2 nao veem (notas na escala, mas erradas); D cai
  D2b         recupera o atraso pelo envelope: atraso_500ms deve dar ~1,0 com atraso ~ +0,5 s (o D2, limitado a +-0,3 s, nao);
              shift_100_atraso_500ms deve cair como o shift_100 (o atraso nao pode compensar o pitch)
  D2          cai em shift_100 (0,67), shift_200 (0,33) e escala_errada (bem abaixo do D), pois nao usa DTW; atraso escolhido = 0
Uso:  python tools/synthetic_check.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import OUT_DIR, ROOT, SR

rng = np.random.default_rng(7)
C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]
SCALE_PC = {0, 2, 4, 5, 7, 9, 11}


def render(midis, durs, seed, atraso=0.0):
    r = np.random.default_rng(seed)
    parts = [np.zeros(int(atraso * SR))] if atraso > 0 else []
    for m, d in zip(midis, durs):
        n = int(d * SR)
        t = np.arange(n) / SR
        f = 440.0 * 2 ** ((m - 69) / 12)
        vib = 1 + (2 ** (20 / 1200) - 1) * np.sin(2 * np.pi * 5.5 * t)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        y = sum(np.sin(k * ph) / k for k in range(1, 7))
        env = np.clip(np.minimum(t / 0.03, (d - t) / 0.03), 0, 1)
        parts += [y * env, np.zeros(int(0.06 * SR))]
    y = np.concatenate(parts)
    y = y + 0.01 * r.standard_normal(len(y))
    return (0.8 * y / np.abs(y).max()).astype(np.float32)


def main():
    base = rng.choice(C_MAJOR, 40).astype(float)
    durs = rng.uniform(0.35, 0.7, 40)
    variants = {
        "perfeita": base,
        "detune_30": base + 0.30,
        "detune_50": base + 0.50,
        "shift_100": base + 1.0,
        "shift_200": base + 2.0,
        "ruido_25c": base + rng.normal(0, 0.25, 40),
        "ruido_50c": base + rng.normal(0, 0.50, 40),
    }
    chrom = base.copy()
    for i in rng.choice(40, 12, replace=False):                      # 30% das notas viram notas FORA da escala
        chrom[i] += -1 if int(chrom[i] + 1) % 12 in SCALE_PC else 1
    variants["cromatica_30pct"] = chrom
    wrong = base.copy()
    for i in rng.choice(40, 20, replace=False):                      # 50% das notas trocadas por OUTRA nota da escala
        wrong[i] = rng.choice([x for x in C_MAJOR if x != base[i]])
    variants["escala_errada_50pct"] = wrong

    atrasadas = {"atraso_500ms": base, "shift_100_atraso_500ms": base + 1.0}     # cantor entra 0,5 s depois da referencia
    out = OUT_DIR / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "referencia.wav", render(base, durs, 1), SR)
    rows = []
    variants = {**variants, **atrasadas}
    for name, m in variants.items():
        sf.write(out / f"{name}.wav", render(m, durs, 2, atraso=0.5 if name in atrasadas else 0.0), SR)
        rows.append(dict(recording=str(out / f"{name}.wav"), tonic="C", mode="major", reference=str(out / "referencia.wav")))
    pd.DataFrame(rows).to_csv(out / "manifesto.csv", index=False)
    labels = list(variants)
    res = subprocess.run([sys.executable, str(ROOT / "13_compare_pitch.py"), "--manifest", str(out / "manifesto.csv")],
                         capture_output=True, text=True, cwd=ROOT)
    print(res.stdout.replace("\n", "\n"))
    if res.returncode:
        print(res.stderr[-1500:])
    print("linhas, na ordem:", labels)


if __name__ == "__main__":
    main()
