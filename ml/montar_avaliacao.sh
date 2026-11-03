#!/usr/bin/env bash
# Monta a pasta avaliacao/ para os ouvintes: covers com codigos (A-F, ordem embaralhada) + trechos dos originais.
# Rode a partir de ml/. NAO mostre o arquivo avaliacao_mapa.csv aos avaliadores.
set -e
mkdir -p avaliacao

cp "dados/covers/hugsPale.mp3" "avaliacao/A.mp3"
cp "dados/covers/intheendLP.mp3" "avaliacao/B.mp3"
cp "dados/covers/só um abraço mesmo pra salvar isso daqui.mp3" "avaliacao/C.mp3"
cp "dados/covers/Chorando em Casa.mp3" "avaliacao/D.mp3"
cp "dados/covers/cover de linkin park.mp3.mp3" "avaliacao/E.mp3"
cp "dados/covers/Ireallywanttostayatyourhouse.mp3" "avaliacao/F.mp3"

# trechos de 30 s dos originais (os mesmos usados nos covers)
ffmpeg -y -loglevel error -ss 53 -t 30 -i "$(ls dados/originais/In*the*End*.mp3 | head -1)" avaliacao/original_In_the_End.mp3
ffmpeg -y -loglevel error -ss 60 -t 30 -i "$(ls dados/originais/Rosa*.mp3 | head -1)" avaliacao/original_I_Really_Want_to_Stay.mp3
ffmpeg -y -loglevel error -ss 46 -t 30 -i "$(ls dados/originais/Paledusk*.mp3 | head -1)" avaliacao/original_Hugs.mp3
echo "pronto: pasta avaliacao/ com $(ls avaliacao | wc -l) arquivos"
