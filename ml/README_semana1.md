# IA do TCC1 — Semana 1 (20–25/10)

Pipeline para o classificador **voz limpa vs. distorcida** com o EMVD. Os scripts foram testados com dados
sintéticos (só para validar que rodam). **Nenhum número desses testes vale como resultado.**

## Ordem de execução

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python 01_download.py                 # baixa o EMVD (Zenodo 8406322) em ./EMVD
python 02_eda.py                     # contagens por classe/cantor, duração, gráficos
python 03_preprocess.py              # janelas de 1 s -> log-Mel 64x100 -> outputs/X.npy
python 04_splits.py                  # folds por cantor e folds estilo artigo
python 05_baseline.py --split singer --task binary
python 05_baseline.py --split singer --task multiclass
python 05_baseline.py --split paper  --task binary      # para medir o vazamento de cantor
python 05_baseline.py --split paper  --task multiclass
```

Para testar antes de baixar o dataset: `python tools/fake_emvd.py` (cria um EMVD falso em ./EMVD; apague depois).

## O que vi no repositório original (modantailleur/ExtremeMetalVocalsDataset)

1. **A divisão do artigo NÃO é por cantor.** `create_split_k_fold.py` usa `KFold(shuffle=True)` sobre as linhas
   de arquivos, então o mesmo cantor pode estar em treino e teste. Por isso os 93% (binário) e 75% (4 classes)
   do artigo provavelmente são otimistas para cantores novos. A sua divisão por cantor deve dar números menores;
   o `--split paper` existe para você mostrar essa diferença no relatório (é um bom resultado por si só).
2. **Classes:** ClearVoice, DeathGrowl, HardcoreScream, BlackShriek. O binário é ClearVoice vs. as outras três.
   O repositório **exclui GrindInhale** e arquivos com `authors_rank == 0` (execução ruim segundo o autor).
3. **Pré-processamento do artigo:** 48 kHz, 128 Mel, n_fft 1024, hop 256, janelas de ~1,02 s sem sobreposição,
   normalização pelo pico por arquivo. **Aqui usei 16 kHz, 64 Mel, hop de 10 ms, janelas de 1 s com hop de 0,5 s
   e normalização por janela**, porque isso é mais leve e mais robusto ao ganho do microfone no navegador.
   Os dois ficam documentados no relatório (decisão de projeto, não erro).
4. **Modelo do artigo:** EfficientNet-B0 (`efficientnet_pytorch`) com 4 splits.
5. O README do repo cita **dois records do Zenodo** (8406322 e 17813678). Veja qual é o mais recente e se os
   metadados mudaram antes de começar.

## Decisões tomadas nos scripts (revise)

- Folds por cantor com `StratifiedGroupKFold` (5 folds, seed 42), verificando que nenhum cantor aparece em treino e teste.
- Janelas quase silenciosas (RMS < 0,01 após normalizar o arquivo) são descartadas.
- Baseline: MFCC, espectro médio/desvio por banda e fluxo espectral; regressão logística e random forest com pesos de classe.
- Métricas: acurácia, acurácia balanceada e F1 macro por janela e por arquivo (média das probabilidades), média ± desvio entre folds.
- Nada foi feito com `Grind Inhale` nem com os "efeitos vocais".

## Checklist da semana 1

- [ ] 01: dataset baixado, `EMVD/audio` e os dois CSVs presentes
- [ ] 02: conferir nomes/valores das colunas (`type`, `name`, `authors_rank`) e se há cantores sem alguma classe
- [ ] 03: `X.npy` gerado; conferir o balanço de janelas por classe
- [ ] 04: nenhum cantor em treino e teste; todas as classes presentes em cada fold de teste
- [ ] 05: tabelas do baseline (singer vs. paper) guardadas para o relatório

## Se algo quebrar

O mais provável é diferença no formato dos CSVs (nomes de coluna, tipo de `authors_rank`). O `02_eda.py` imprime
as colunas e os valores únicos. Me mande essa saída e eu ajusto o `common.py`.
