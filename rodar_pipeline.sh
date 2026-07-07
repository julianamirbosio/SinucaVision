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
#   04_detector_bolas  → detecta bolas (YOLO) e gera JSONs em data/output/<nome>_balls.json
#   06_feature_eng     → reconstrói data/dataset.csv
#   07_treino_rf       → treina o modelo e salva models/random_forest_jogada.pkl
#
# PRÉ-REQUISITO ADICIONAL: models/yolo_deteccao_bolas.pt já treinado
#   (ver "Como criar os modelos" no README — este script não treina o YOLO)
#
# NOTEBOOKS DE USO PONTUAL (rode manualmente quando quiser):
#   05_visualizador    → debug de detecções
#   08_inferencia      → recomenda jogada para uma imagem específica

set -e
cd "$(dirname "$0")"

JUPYTER="jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    --ExecutePreprocessor.kernel_name=sinucavision"

echo "=== SinucaVision Pipeline ==="
echo ""

echo "[1/3] Detector de bolas..."
$JUPYTER modulo1/04_detector_bolas.ipynb
echo "      OK"

echo "[2/3] Engenharia de features..."
$JUPYTER modulo2/06_feature_eng.ipynb
echo "      OK"

echo "[3/3] Treino do Random Forest..."
$JUPYTER modulo3/07_treino_rf.ipynb
echo "      OK"

echo ""
echo "=== Concluído. Modelo salvo em models/random_forest_jogada.pkl ==="
