#!/bin/bash
# Pipeline de treino do SinucaVision.
#
# PRÉ-REQUISITOS MANUAIS (faça antes de rodar este script):
#   1. Coloque as fotos originais em data/raw/
#   2. Abra 00_converter_heic.ipynb e rode manualmente (converte HEIC → JPG)
#   3. Abra 01_homografia.ipynb e anote as 4 quinas de cada imagem
#      (clique interativo — inviável automatizar)
#
# ESTE SCRIPT RODA:
#   02_detector_bolas  → detecta bolas e gera JSONs em data/output/<nome>/
#   04_feature_eng     → reconstrói data/dataset.csv
#   05_treino_rf       → treina o modelo e salva models/modelo.pkl
#
# NOTEBOOKS DE USO PONTUAL (rode manualmente quando quiser):
#   03_visualizador    → debug de detecções
#   06_inferencia      → recomenda jogada para uma imagem específica

set -e
cd "$(dirname "$0")"

JUPYTER="jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    --ExecutePreprocessor.kernel_name=sinucavision"

echo "=== SinucaVision Pipeline ==="
echo ""

echo "[1/3] Detector de bolas..."
$JUPYTER modulo1/02_detector_bolas.ipynb
echo "      OK"

echo "[2/3] Engenharia de features..."
$JUPYTER modulo2/04_feature_eng.ipynb
echo "      OK"

echo "[3/3] Treino do Random Forest..."
$JUPYTER modulo3/05_treino_rf.ipynb
echo "      OK"

echo ""
echo "=== Concluído. Modelo salvo em models/modelo.pkl ==="
