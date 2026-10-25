"""Audio -> janelas de 1 s (hop 0,5 s) -> log-Mel (64 x 100). Gera outputs/X.npy e windows.csv."""
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import *


def logmel(w: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    db = 10 * np.log10(m + 1e-10)
    x = np.clip((db + 100) / 100, 0, None)  # mesma escala usada no repositorio original
    return x[:, :N_FRAMES]


def windows(y: np.ndarray):
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    y = y / (np.abs(y).max() + 1e-9)
    for s in range(0, len(y) - win + 1, hop):
        w = y[s:s + win]
        rms = float(np.sqrt((w ** 2).mean()))
        if rms < SILENCE_RMS:
            continue
        yield s / SR, w * (TARGET_RMS / (rms + 1e-9))  # normaliza por janela


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_metadata(verbose=True)
    X, rows, curtos = [], [], 0
    for r in tqdm(df.itertuples(index=False), total=len(df)):
        y, _ = librosa.load(AUDIO_DIR / r.file_name, sr=SR, mono=True)
        got = 0
        for start, w in windows(y):
            X.append(logmel(w).astype(np.float16))
            rows.append((r.file_name, r.singer_id, r.name, r.y4, r.ybin, round(start, 2)))
            got += 1
        curtos += got == 0
    X = np.stack(X)
    meta = pd.DataFrame(rows, columns=["file_name", "singer_id", "name", "y4", "ybin", "start_s"])
    np.save(OUT_DIR / "X.npy", X)
    meta.to_csv(OUT_DIR / "windows.csv", index=False)
    print(f"X: {X.shape} ({X.nbytes / 1e6:.0f} MB) | arquivos sem janela valida: {curtos}")
    print(meta.groupby("name").size().reindex(CLASSES).to_string())


if __name__ == "__main__":
    main()
