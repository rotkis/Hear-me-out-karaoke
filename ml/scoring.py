"""Tres metodos para a componente de PITCH do score (todos usam a confianca do YIN como peso por quadro).

A  - grade cromatica absoluta (familia Nakano 2006), referencia A4 = 440 Hz; NAO desconta o desvio do cantor.
B1 - tonalidade, binario (Hsieh et al., 2025): proporcao de quadros cuja nota cai na escala da tonalidade.
B2 - tonalidade, continuo (variante NOSSA, nao avaliada na literatura): distancia ate a nota da escala mais proxima.
D  - melodia extraida da faixa (voz separada): similaridade de classe de altura com a voz do cantor, com DTW em faixa.
D2 - como o D, mas SEM DTW: comparacao quadro a quadro com um unico atraso global (preserva a ordem da melodia).
D2b- como o D2, mas o atraso vem do ENVELOPE DE ENERGIA (quando a voz entra e sai), nao do pitch; depois so compara pitch nesse atraso.

Todos retornam valores em [0, 1] (mais alto = melhor). Niveis de acaso aproximados: A 0,5; B1 ~0,58; B2 ~0,5; D ~0.
"""
import numpy as np

RAMP_LO, RAMP_HI = 0.10, 0.20     # peso do pitch: 1 ate CMNDF 0,10, cai linearmente a 0 em 0,20 (SDD 11.1; a calibrar)
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def weights(tr):
    w = np.clip((RAMP_HI - tr["cmndf"]) / (RAMP_HI - RAMP_LO), 0.0, 1.0)
    return w * tr["active"]


def midi(f0, a4=440.0):
    return 69.0 + 12.0 * np.log2(f0 / a4)


def _wmean(s, w):
    tot = float(w.sum())
    return float((s * w).sum() / tot) if tot > 0 else float("nan")


def score_grid(tr, a4=440.0):
    w, m = weights(tr), midi(tr["f0"], a4)
    dev = np.abs(m - np.round(m))                      # 0..0,5 semitom
    return _wmean(1.0 - dev / 0.5, w)


def score_key(tr, tonic, mode, a4=440.0, continuous=False):
    w, m = weights(tr), midi(tr["f0"], a4)
    if not continuous:
        scale = (np.array(SCALES[mode]) + tonic) % 12
        s = np.isin(np.round(m).astype(int) % 12, scale).astype(float)
    else:
        rel = (m - tonic) % 12.0
        d = np.abs(rel[:, None] - np.array(SCALES[mode])[None, :])
        d = np.minimum(d, 12.0 - d).min(axis=1)        # semitons ate a nota da escala mais proxima
        s = 1.0 - np.minimum(d, 1.0)
    return _wmean(s, w)


def _pc_contour(tr, a4=440.0, min_w=0.5, rate=25):
    """Classe de altura (0..12) por bin de 1/rate s (media circular dos quadros confiaveis); NaN onde nao ha."""
    w, m = weights(tr), midi(tr["f0"], a4)
    ang = 2 * np.pi * (m % 12.0) / 12.0
    ok = w >= min_w
    nb = int(np.ceil(tr["t"][-1] * rate)) + 1 if len(tr["t"]) else 0
    b = (tr["t"] * rate).astype(int)
    out = np.full(nb, np.nan)
    for i in np.unique(b[ok]):
        sel = ok & (b == i)
        z = np.exp(1j * ang[sel]).mean()
        if abs(z) > 0.3:
            out[i] = (np.angle(z) % (2 * np.pi)) * 12.0 / (2 * np.pi)
    return out


def score_ref(tr_user, tr_ref, a4=440.0, band=0.1, min_frames=10):
    """D: DTW (com faixa) entre as classes de altura do cantor e da voz de referencia; so quadros confiaveis nos dois.

    band = largura da faixa do DTW (fracao do tamanho da sequencia). Faixa larga deixa o DTW casar qualquer nota com outra de mesma
    classe (perde a ordem da melodia); faixa estreita pune mais, mas tolera menos atraso. Como a reproducao e sincronizada, comece
    estreito e alargue so se gravacoes reais mostrarem atrasos grandes."""
    import librosa

    u = _pc_contour(tr_user, a4)
    r = _pc_contour(tr_ref, a4)
    u, r = u[~np.isnan(u)], r[~np.isnan(r)]
    if len(u) < min_frames or len(r) < min_frames:
        return float("nan")
    diff = np.abs(u[:, None] - r[None, :])
    C = np.minimum(diff, 12.0 - diff)                  # distancia circular em semitons (0..6)
    D, wp = librosa.sequence.dtw(C=C, global_constraints=True, band_rad=band)
    if not np.isfinite(D[-1, -1]):
        return float("nan")
    mean_cost = float(D[-1, -1] / len(wp))
    return float(np.clip(1.0 - mean_cost / 3.0, 0.0, 1.0))   # 3 semitons = custo medio de classes de altura ao acaso


def ref_lag_detail(tr_user, tr_ref, a4=440.0, max_lag_s=0.3, rate=25, min_pairs=30, min_cobertura=0.10):
    """D2: classes de altura do cantor x da voz de referencia, bin a bin (1/rate s), com UM atraso global.

    O atraso L (em bins, de -max_lag_s a +max_lag_s) compara u[i] com r[i-L]; L > 0 = o cantor entra depois da referencia.
    Entre os atrasos, so concorrem os que tem pares confiaveis suficientes: >= max(min_pairs, 50% do maximo de pares entre
    todos os atrasos). Sem isso, um atraso com 2 ou 3 pares poderia ganhar por acaso. Vence o de menor custo medio
    (distancia circular de classe de altura, em semitons). score = clip(1 - custo/3, 0, 1), igual ao D.
    cobertura = pares no atraso escolhido / bins confiaveis do cantor. Se cobertura < min_cobertura, score = NaN.
    Limiares (min_pairs=30 = 1,2 s; min_cobertura=0,10) sao CONSERVADORES e arbitrarios: ajustar vendo a cobertura real.
    """
    u = _pc_contour(tr_user, a4, rate=rate)
    r = _pc_contour(tr_ref, a4, rate=rate)
    out = dict(score=float("nan"), atraso_s=float("nan"), cobertura=float("nan"), custo=float("nan"), pares=0)
    n_user = int(np.sum(~np.isnan(u)))
    if n_user == 0 or np.all(np.isnan(r)):
        return out
    max_lag = int(round(max_lag_s * rate))
    cands = []
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            a, b = u[L:], r[: max(len(r) - L, 0)]
        else:
            a, b = u[: max(len(u) + L, 0)], r[-L:]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() == 0:
            continue
        d = np.abs(a[ok] - b[ok])
        cands.append((L, int(ok.sum()), float(np.minimum(d, 12.0 - d).mean())))
    if not cands:
        return out
    max_pairs = max(c[1] for c in cands)
    elig = [c for c in cands if c[1] >= max(min_pairs, 0.5 * max_pairs)]
    if not elig:
        return out
    L, n, cost = min(elig, key=lambda c: c[2])
    cob = n / n_user
    out.update(atraso_s=L / rate, cobertura=cob, custo=cost, pares=n)
    if cob >= min_cobertura:
        out["score"] = float(np.clip(1.0 - cost / 3.0, 0.0, 1.0))
    return out


def score_ref_lag(tr_user, tr_ref, **kw):
    return ref_lag_detail(tr_user, tr_ref, **kw)["score"]


def _envelope(tr, rate=25):
    """Log-energia da voz por bin de 1/rate s, com piso 40 dB abaixo do pico e z-score. Nao depende de pitch."""
    per = int(round(100 / rate))                       # quadros de 10 ms por bin
    r = np.log10(np.asarray(tr["rms"], dtype=float) + 1e-4)
    nb = len(r) // per
    if nb < 4:
        return np.zeros(0)
    e = r[: nb * per].reshape(nb, per).mean(axis=1)
    e = np.clip(e, e.max() - 2.0, None)
    return (e - e.mean()) / (e.std() + 1e-9)


def _overlap(x, y, L):
    """Trechos de x e y alinhados para o atraso L (L > 0: x entra depois de y): x[i] <-> y[i-L]."""
    if L >= 0:
        a, b = x[L:], y[: max(len(y) - L, 0)]
    else:
        a, b = x[: max(len(x) + L, 0)], y[-L:]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def ref_env_detail(tr_user, tr_ref, a4=440.0, max_lag_s=1.0, rate=25, min_corr=0.10, min_pairs=30, min_cobertura=0.10):
    """D2b: 1) acha o atraso global pelo envelope de energia (correlacao cruzada em +-max_lag_s, sem olhar o pitch);
    2) nesse atraso FIXO compara as classes de altura bin a bin. Como o atraso nao depende do pitch, ele nao pode ser usado
    para 'compensar' uma nota deslocada (a brecha possivel do D2). score = clip(1 - custo/3, 0, 1), igual ao D e ao D2.
    Devolve tambem atraso_s, corr_env (correlacao do envelope no atraso: confianca do alinhamento), cobertura e custo.
    NaN se corr_env < min_corr, se houver menos de min_pairs pares ou cobertura < min_cobertura (limiares arbitrarios)."""
    out = dict(score=float("nan"), atraso_s=float("nan"), corr_env=float("nan"), cobertura=float("nan"), custo=float("nan"), pares=0)
    eu, er = _envelope(tr_user, rate), _envelope(tr_ref, rate)
    if len(eu) < 10 or len(er) < 10:
        return out
    max_lag = int(round(max_lag_s * rate))
    best = None
    for L in range(-max_lag, max_lag + 1):
        a, b = _overlap(eu, er, L)
        if len(a) < max(20, 0.5 * min(len(eu), len(er))):      # exige sobreposicao de pelo menos metade do trecho
            continue
        c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        if np.isfinite(c) and (best is None or c > best[1]):
            best = (L, c)
    if best is None:
        return out
    L, corr = best
    out.update(atraso_s=L / rate, corr_env=corr)
    u = _pc_contour(tr_user, a4, rate=rate)
    r = _pc_contour(tr_ref, a4, rate=rate)
    n_user = int(np.sum(~np.isnan(u)))
    a, b = _overlap(u, r, L)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if n_user == 0 or ok.sum() == 0:
        return out
    d = np.abs(a[ok] - b[ok])
    cost = float(np.minimum(d, 12.0 - d).mean())
    cob = int(ok.sum()) / n_user
    out.update(custo=cost, cobertura=cob, pares=int(ok.sum()))
    if corr >= min_corr and ok.sum() >= min_pairs and cob >= min_cobertura:
        out["score"] = float(np.clip(1.0 - cost / 3.0, 0.0, 1.0))
    return out


def score_ref_env(tr_user, tr_ref, **kw):
    return ref_env_detail(tr_user, tr_ref, **kw)["score"]
