"""Gera um EMVD FALSO (audio sintetico) para testar o pipeline antes de baixar o dataset real.
Uso: python tools/fake_emvd.py   (cria ./EMVD/audio e os metadata). NAO use esses dados para resultados."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import AUDIO_DIR, CLASSES, EMVD_DIR

rng = np.random.default_rng(0)
SR = 48000


def synth(cls: str, dur: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    f0 = rng.uniform(110, 330) if cls == "ClearVoice" else rng.uniform(60, 200)
    vib = 1 + 0.01 * np.sin(2 * np.pi * 5.5 * t)
    ph = 2 * np.pi * np.cumsum(f0 * vib) / SR
    y = sum(np.sin(k * ph) / k for k in range(1, 8))
    noise = {"ClearVoice": 0.02, "DeathGrowl": 0.6, "HardcoreScream": 0.9, "BlackShriek": 1.2}[cls]
    y = y + noise * rng.standard_normal(len(t)) * (1 + 0.5 * np.sin(2 * np.pi * 30 * t))
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(1, 13):
        for c in CLASSES:
            for i in range(3):
                dur = float(rng.uniform(4, 7))
                name = f"s{s:02d}_{c}_{i}.wav"
                sf.write(AUDIO_DIR / name, synth(c, dur), SR)
                rows.append(dict(file_name=name, singer_id=s, type="Technique", name=c, range="mid", vowel="a", authors_rank=int(rng.integers(1, 3)), **{"duration(s)": dur}))
    pd.DataFrame(rows).to_csv(EMVD_DIR / "metadata_files.csv", index=False)
    pd.DataFrame(dict(singer_id=range(1, 13), country="XX")).to_csv(EMVD_DIR / "metadata_singers.csv", index=False)
    print("EMVD falso criado:", len(rows), "arquivos")


if __name__ == "__main__":
    main()
