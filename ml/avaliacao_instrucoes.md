# Avaliação humana de afinação dos 6 covers

Objetivo: ter uma régua humana para saber qual método de pontuação (A, B1, B2, D, D2, D2b) concorda com o que as pessoas ouvem.

## Preparação (você)
1. Rode a partir de `ml/`: `bash montar_avaliacao.sh` (cria a pasta `avaliacao/` com os covers `A.mp3` a `F.mp3`, em ordem embaralhada, e os trechos de 30 s dos 3 originais).
2. **Não mostre** `avaliacao_mapa.csv` nem os resultados dos métodos aos avaliadores.
3. Combine quantas pessoas vão avaliar. Quanto mais, melhor. Você e seu amigo são os cantores e sabem quem cantou o quê; inclua pelo menos duas pessoas de fora.

## Para quem avalia
- Ouça primeiro o trecho do original da música, depois o cover (cada cover tem a música dele; o código A a F não diz quem cantou).
- Pergunta: **"quão afinado está o cantor em relação à música original?"** Nota de 1 (muito desafinado) a 5 (muito afinado).
- Avalie só a afinação. Ignore timbre, emoção, técnica, qualidade da gravação e a língua da letra.
- Em trechos gritados ou guturais, julgue se a melodia (para onde a voz sobe e desce) acompanha a música.
- Avalie cada cover sozinho(a), sem conversar com os outros antes de terminar.

## Registro
1. Copie as 6 linhas de `avaliacoes_template.csv` para cada avaliador e preencha `avaliador` e `nota_afinacao_1a5`. Salve tudo como `avaliacoes.csv`.
2. Rode:
   `python tools/aggregate_ratings.py --ratings avaliacoes.csv --mapa avaliacao_mapa.csv --manifest manifesto_covers.csv`
   Isso cria `manifesto_com_notas.csv` (média por cover na coluna `rating`) e mostra a concordância entre avaliadores.
3. Compare os métodos com as notas: `python 13_compare_pitch.py --manifest manifesto_com_notas.csv`

## Limites para registrar no relatório
- Com 6 covers, qualquer correlação é anedótica (só valores acima de cerca de 0,89 seriam significativos a 5%).
- Se os avaliadores discordarem muito entre si (concordância baixa), a nota humana é pouco confiável e isso limita qualquer conclusão sobre os métodos.
