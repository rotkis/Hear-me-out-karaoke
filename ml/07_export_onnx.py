"""Treina o modelo FINAL (todos os cantores) e exporta para o navegador.

Gera em web_test/:
  model.onnx          entrada: log-Mel (1,1,64,100) na escala do treino; saida: probabilidades (softmax).
                      A normalizacao (media/desvio) ja esta dentro do modelo.
  model_meta.json     parametros do pre-processamento + banco de filtros Mel do librosa + padding da STFT.
  test_vectors.json   janelas de audio reais, o log-Mel esperado e as probabilidades esperadas (teste de paridade).

Uso:
    pip install onnx onnxruntime
    python 07_export_onnx.py --task binary
    python 07_export_onnx.py --task binary --no-aug      # (ou o que vencer o teste de robustez)
Depois:  cd web_test && python -m http.server 8000   e abra http://localhost:8000
"""
import argparse
import importlib.util
import inspect
import json

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from common import AUDIO_DIR, CLASSES, FMAX, FMIN, HOP, HOP_S, N_FFT, N_FRAMES, N_MELS, OUT_DIR, ROOT, SEED, SILENCE_RMS, SR, TARGET_RMS, WIN_S, load_metadata

WEB_DIR = ROOT / "web_test"


def _load(file: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cnn = _load("06_cnn.py", "cnn06")
pre = _load("03_preprocess.py", "pre03")


class Deploy(nn.Module):
    """CNN + normalizacao + softmax: o navegador so precisa fornecer o log-Mel."""

    def __init__(self, net: nn.Module, mean: float, std: float):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, x):
        return torch.softmax(self.net((x - self.mean) / self.std), dim=1)


def main(a):
    WEB_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 2 if a.task == "binary" else 4
    ycol = "ybin" if a.task == "binary" else "y4"

    X = torch.from_numpy(np.load(OUT_DIR / "X.npy").astype(np.float32)).unsqueeze(1)
    w = pd.read_csv(OUT_DIR / "windows.csv")
    y = w[ycol].to_numpy().astype(np.int64)

    # Treino final: todos os cantores; validacao (parada antecipada) com ~15% dos cantores.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    fit_idx, val_idx = next(gss.split(np.arange(len(w)), groups=w["singer_id"].to_numpy()))
    targs = argparse.Namespace(lr=1e-3, epochs=a.epochs, patience=a.patience, batch=128, no_aug=a.no_aug)
    model, mean, std, best_val, n_ep = cnn.train_fold(X, y, fit_idx, val_idx, n_classes, targs, device, SEED)
    print(f"modelo final: {n_ep} epocas, val_bal_acc={best_val:.3f}, media={mean:.4f}, desvio={std:.4f}")

    dep = Deploy(model.cpu(), mean, std).eval()
    out = WEB_DIR / "model.onnx"
    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES)
    kw = dict(input_names=["logmel"], output_names=["probs"], opset_version=17)
    try:
        torch.onnx.export(dep, dummy, str(out), dynamo=False, **kw)   # exportador classico: um unico arquivo .onnx
    except TypeError:                                                  # versoes antigas do torch nao tem o argumento
        torch.onnx.export(dep, dummy, str(out), **kw)
    print("exportado:", out, f"({out.stat().st_size / 1024:.0f} KB)")

    # --- paridade PyTorch x ONNX Runtime (Python) ---
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    idx = np.random.default_rng(0).choice(len(X), 64, replace=False)
    with torch.no_grad():
        ref = dep(X[idx]).numpy()
    got = np.concatenate([sess.run(None, {"logmel": X[i:i + 1].numpy()})[0] for i in idx])
    diff = float(np.abs(ref - got).max())
    print(f"paridade torch x onnx: erro maximo = {diff:.2e}", "OK" if diff < 1e-4 else "<<< ATENCAO")

    # --- metadados para o navegador ---
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    pad_mode = inspect.signature(librosa.stft).parameters["pad_mode"].default
    meta = dict(
        task=a.task, classes=["limpa", "distorcida"] if a.task == "binary" else CLASSES,
        sr=SR, nFft=N_FFT, hop=HOP, nMels=N_MELS, nFrames=N_FRAMES, fmin=FMIN, fmax=FMAX,
        winS=WIN_S, hopS=HOP_S, silenceRms=SILENCE_RMS, targetRms=TARGET_RMS, padMode=pad_mode,
        trainedWithAugmentation=not a.no_aug, valBalAcc=round(float(best_val), 4), epochs=n_ep,
        filterbank=np.round(fb, 7).tolist(),
    )
    (WEB_DIR / "model_meta.json").write_text(json.dumps(meta))

    # --- vetores de teste: 1 janela real de cada classe ---
    df = load_metadata()
    vectors = []
    for cls in CLASSES:
        row = df[df["name"] == cls].iloc[0]
        audio, _ = librosa.load(AUDIO_DIR / row["file_name"], sr=SR, mono=True)
        _, win = next(pre.windows(audio))
        win = np.round(win, 6).astype(np.float32)           # o JS recebe exatamente estes valores
        lm = pre.logmel(win).astype(np.float32)
        p = sess.run(None, {"logmel": lm[None, None]})[0][0]
        vectors.append(dict(name=cls, label=int(row["ybin"] if a.task == "binary" else row["y4"]),
                            audio=win.tolist(), logmel=lm.ravel().tolist(), probs=[float(v) for v in p]))
    (WEB_DIR / "test_vectors.json").write_text(json.dumps(vectors))
    print("pad_mode da STFT no seu librosa:", pad_mode, "(vai em model_meta.json e o JS usa o mesmo)")
    print("\nProximo passo:  cd web_test && python -m http.server 8000   ->   http://localhost:8000")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--no-aug", action="store_true")
    main(p.parse_args())
