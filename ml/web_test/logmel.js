/* Log-Mel identico ao 03_preprocess.py (librosa): STFT Hann periodica, potencia, filtros Mel exportados do librosa,
 * 10*log10(m + 1e-10), (db + 100) / 100, piso em 0. Funciona no navegador (window.LogMel) e no Node (require). */
(function (root) {
  'use strict';

  function makeLogMel(cfg) {
    // cfg: {nFft, hop, nMels, nFrames, padMode: 'constant' | 'reflect', filterbank: number[nMels][nFft/2+1]}
    const N = cfg.nFft, H = cfg.hop, M = cfg.nMels, F = cfg.nFrames, bins = N / 2 + 1;
    const padMode = cfg.padMode || 'constant';

    const win = new Float64Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N); // Hann periodica

    const cosT = new Float64Array(N / 2), sinT = new Float64Array(N / 2);
    for (let i = 0; i < N / 2; i++) { cosT[i] = Math.cos((2 * Math.PI * i) / N); sinT[i] = Math.sin((2 * Math.PI * i) / N); }
    const rev = new Uint32Array(N);
    const bitsN = Math.log2(N) | 0;
    for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < bitsN; b++) if (i & (1 << b)) r |= 1 << (bitsN - 1 - b); rev[i] = r; }

    const fb = new Float64Array(M * bins);
    for (let m = 0; m < M; m++) for (let k = 0; k < bins; k++) fb[m * bins + k] = cfg.filterbank[m][k];

    const re = new Float64Array(N), im = new Float64Array(N), power = new Float64Array(bins);

    function fft() {
      for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
      for (let size = 2; size <= N; size <<= 1) {
        const half = size >> 1, step = N / size;
        for (let start = 0; start < N; start += size) {
          for (let k = 0, tw = 0; k < half; k++, tw += step) {
            const a = start + k, b = a + half;
            const tr = re[b] * cosT[tw] + im[b] * sinT[tw];
            const ti = -re[b] * sinT[tw] + im[b] * cosT[tw];
            re[b] = re[a] - tr; im[b] = im[a] - ti;
            re[a] += tr; im[a] += ti;
          }
        }
      }
    }

    function sample(y, idx) {
      const L = y.length;
      if (idx >= 0 && idx < L) return y[idx];
      if (padMode === 'constant') return 0;
      let i = idx < 0 ? -idx : idx;               // reflect
      if (i >= L) i = 2 * (L - 1) - i;
      return y[i];
    }

    // y: Float32Array (ex.: 16000 amostras). Retorna Float32Array(nMels * nFrames), indice m * nFrames + t.
    return function logmel(y) {
      const out = new Float32Array(M * F);
      for (let t = 0; t < F; t++) {
        const start = t * H - N / 2;
        for (let n = 0; n < N; n++) { re[n] = sample(y, start + n) * win[n]; im[n] = 0; }
        fft();
        for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
        for (let m = 0; m < M; m++) {
          let s = 0; const base = m * bins;
          for (let k = 0; k < bins; k++) s += fb[base + k] * power[k];
          const v = (10 * Math.log10(s + 1e-10) + 100) / 100;
          out[m * F + t] = v > 0 ? v : 0;
        }
      }
      return out;
    };
  }

  // Normaliza uma janela como no treino: RMS -> targetRms. Retorna {y, rms} (rms ANTES da normalizacao).
  function normalizeWindow(y, targetRms) {
    let s = 0;
    for (let i = 0; i < y.length; i++) s += y[i] * y[i];
    const rms = Math.sqrt(s / y.length);
    const g = targetRms / (rms + 1e-9);
    const out = new Float32Array(y.length);
    for (let i = 0; i < y.length; i++) out[i] = y[i] * g;
    return { y: out, rms };
  }

  const api = { makeLogMel, normalizeWindow };
  root.LogMel = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : this);
