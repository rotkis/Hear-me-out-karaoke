"""Separa a voz de uma musica (Demucs) para servir de referencia ao metodo D, opcionalmente so de um trecho.

Pre-requisito:  pip install demucs        (usa o PyTorch que voce ja tem)
Uso:
    python 12_reference_vocal.py musica.mp3 --start 53 --end 83 --name LP
Saida:  outputs/stems/htdemucs/<name>/vocals.wav   (a voz do trecho, comecando em 0 s)
Cortar o trecho antes de separar deixa o Demucs bem mais rapido e alinha a referencia ao seu cover de ~30 s.
Use apenas arquivos que voce tenha licenca para usar (nao baixe do YouTube).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import librosa
import soundfile as sf

from common import OUT_DIR


def trim(path: str, start: float, end: float, name: str) -> Path:
    y, sr = librosa.load(path, sr=None, mono=False, offset=start, duration=end - start)
    out = OUT_DIR / "stems" / "_trecho" / f"{name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y.T if y.ndim > 1 else y, sr)
    return out


def main(a):
    src = Path(a.audio).expanduser()
    if not src.exists():
        raise SystemExit(f"arquivo nao encontrado: {src.resolve()}\nInforme o caminho completo ou coloque o arquivo em ml/dados/originais/ e use esse caminho.")
    a.audio = str(src)
    if a.start is not None or a.end is not None:
        if a.start is None or a.end is None or a.end <= a.start:
            raise SystemExit("informe --start e --end (segundos), com end > start")
        name = a.name or f"{src.stem}_{a.start:g}-{a.end:g}"
        src = trim(a.audio, a.start, a.end, name)
    else:
        name = a.name or src.stem
        if a.name:
            tmp = OUT_DIR / "stems" / "_trecho" / f"{name}{src.suffix}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(src.read_bytes())
            src = tmp
    out = OUT_DIR / "stems"
    subprocess.run([sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "--two-stems=vocals", "-o", str(out), str(src)], check=True)
    voc = out / "htdemucs" / src.stem / "vocals.wav"
    print("voz separada:", voc, "(existe)" if voc.exists() else "(NAO encontrado: confira a saida do demucs)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--name", help="nome curto da saida (ex.: LP)")
    main(p.parse_args())
