"""Estimador de pitch YIN com interpolacao parabolica + confianca (minimo do CMNDF).

A interpolacao importa: a 16 kHz, um passo inteiro de atraso vale ~48 cents em 440 Hz.
Retorna, por quadro: f0 (Hz), cmndf (quanto menor, mais periodico), rms e a mascara de quadros ativos.
"""
import numpy as np

from common import SR

FRAME, HOP = 1024, 160            # janela de 64 ms, passo de 10 ms
FMIN, FMAX = 65.0, 1000.0
ACTIVE_RMS = 0.02                 # com o sinal normalizado pelo pico
ABS_THRESHOLD = 0.15              # limiar absoluto do YIN para escolher o atraso (evita subharmonicos)


def track(y, sr=SR, interpolate=True):
    y = np.asarray(y, dtype=np.float64)
    y = y / (np.abs(y).max() + 1e-12)
    tau_min, tau_max = int(sr / FMAX), int(sr / FMIN)
    W = FRAME // 2
    n = 1 + (len(y) - FRAME) // HOP
    if n <= 0:
        e = np.empty(0)
        return dict(t=e, f0=e, cmndf=e, rms=e, active=e.astype(bool))
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    F = y[idx]
    rms = np.sqrt((F ** 2).mean(axis=1))

    nfft = 2048
    A = np.fft.rfft(F[:, :W], nfft, axis=1)
    B = np.fft.rfft(F, nfft, axis=1)
    c = np.fft.irfft(np.conj(A) * B, nfft, axis=1)[:, : tau_max + 2]
    e_a = (F[:, :W] ** 2).sum(axis=1, keepdims=True)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(F ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 2)
    e_b = cs[:, taus + W] - cs[:, taus]
    d = np.maximum(e_a + e_b - 2 * c, 0)
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1)
    cm = np.ones_like(d)
    cm[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-12)

    seg = cm[:, tau_min: tau_max + 2]                  # um atraso extra para vizinho na interpolacao
    search = seg[:, : tau_max - tau_min + 1]
    below = search < ABS_THRESHOLD
    first = np.where(below.any(axis=1), below.argmax(axis=1), search.argmin(axis=1))
    win = np.minimum(first[:, None] + np.arange(12)[None, :], search.shape[1] - 1)
    rows = np.arange(n)
    k = win[rows, np.take_along_axis(search, win, axis=1).argmin(axis=1)]

    tau = (tau_min + k).astype(np.float64)
    if interpolate:
        y0 = seg[rows, k]
        ym = seg[rows, np.clip(k - 1, 0, seg.shape[1] - 1)]
        yp = seg[rows, np.clip(k + 1, 0, seg.shape[1] - 1)]
        den = ym - 2 * y0 + yp
        delta = np.where(den > 1e-9, 0.5 * (ym - yp) / np.where(den > 1e-9, den, 1), 0.0)
        tau += np.clip(delta, -1, 1)
    return dict(
        t=(FRAME / 2 + HOP * np.arange(n)) / sr,
        f0=sr / tau,
        cmndf=search.min(axis=1),
        rms=rms,
        active=rms > ACTIVE_RMS,
    )
