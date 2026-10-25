"""Constantes e utilitarios compartilhados (semana 1 - IA do TCC)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
EMVD_DIR = ROOT / "EMVD"
AUDIO_DIR = EMVD_DIR / "audio"
OUT_DIR = ROOT / "outputs"

# Mesmas classes do repositorio do EMVD (Tailleur et al., 2024).
CLASSES = ["ClearVoice", "DeathGrowl", "HardcoreScream", "BlackShriek"]
EXCLUDE_NAMES = {"GrindInhale"}  # o repositorio original tambem exclui

# Pre-processamento pensado para rodar depois no navegador (leve):
SR = 16000
N_FFT = 512
HOP = 160            # 10 ms
N_MELS = 64
FMIN, FMAX = 20, 8000
WIN_S, HOP_S = 1.0, 0.5
N_FRAMES = int(WIN_S * SR / HOP)  # 100
SILENCE_RMS = 0.01   # descarta janelas quase silenciosas (apos normalizar o arquivo pelo pico)
TARGET_RMS = 0.1     # normalizacao por janela (robusta a ganho do microfone)
SEED = 42
def _durations(df: pd.DataFrame) -> pd.Series:
    """Duracao em segundos. Tenta o CSV; o que nao for numerico vem do cabecalho do audio."""
    import soundfile as sf

    col = "duration(s)"
    if col in df.columns:
        txt = df[col].astype(str).str.replace(",", ".", regex=False).str.extract(r"([0-9]+\.?[0-9]*)")[0]
        d = pd.to_numeric(txt, errors="coerce")
    else:
        d = pd.Series(float("nan"), index=df.index)
    for i in d.index[d.isna()]:
        try:
            d[i] = sf.info(str(AUDIO_DIR / df.at[i, "file_name"])).duration
        except Exception:
            pass
    return d

def load_metadata(keep_rank0: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Le metadata_files.csv aplicando os mesmos filtros do repositorio original."""
    df = pd.read_csv(EMVD_DIR / "metadata_files.csv")
    n0 = len(df)
    df = df[df["type"] == "Technique"]
    df = df[df["name"].isin(CLASSES) & ~df["name"].isin(EXCLUDE_NAMES)]
    n1 = len(df)
    if not keep_rank0 and "authors_rank" in df.columns:
        rank = pd.to_numeric(df["authors_rank"], errors="coerce")
        df = df[~(rank == 0)]  # rank 0 = execucao ruim segundo o autor do dataset
    if verbose:
        print(f"[metadata] {n0} linhas -> {n1} (tecnicas/classes) -> {len(df)} (sem rank 0)")
    df = df.reset_index(drop=True)
    df["duration(s)"] = _durations(df)
    sp = EMVD_DIR / "metadata_singers.csv"
    if sp.exists():
        singers = pd.read_csv(sp)
        if "singer_id" in singers.columns:
            df = df.merge(singers, on="singer_id", how="left", suffixes=("", "_singer"))
    df["y4"] = df["name"].map({c: i for i, c in enumerate(CLASSES)})
    df["ybin"] = (df["name"] != "ClearVoice").astype(int)  # 1 = distorcida
    return df
