#!/usr/bin/env python3
"""
Pega uma foto crua da mesa de sinuca e devolve a jogada sugerida.

Uso:
    python inferir_jogada.py caminho/para/foto.jpg --time lisa

Etapas (equivalentes aos notebooks 00, 01, 04 e 08):
  1. Converte a foto para JPG/RGB, se necessário (HEIC, PNG, etc.)
  2. Pede para clicar as 4 quinas da mesa (homografia -> vista de cima 800x400)
  3. Detecta as bolas e caçapas com o modelo YOLO treinado
  4. Monta o vetor de features geométricas
  5. Aplica as regras do 8-ball e o Random Forest treinado
  6. Imprime a jogada sugerida e salva uma imagem com a visualização
"""

import argparse
import math
import os
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from pillow_heif import register_heif_opener
from PIL import Image
from ultralytics import YOLO

register_heif_opener()

RAIZ = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO_YOLO = os.path.join(RAIZ, "models", "yolo_deteccao_bolas.pt")
CAMINHO_MODELO_RF   = os.path.join(RAIZ, "models", "random_forest_jogada.pkl")
CAMINHO_RESULTADOS  = os.path.join(RAIZ, "data", "inferir-jogada-results")

LARGURA_MESA, ALTURA_MESA = 800, 400
NOMES_CLASSES = {0: "branca", 1: "lisa", 2: "listrada", 3: "preta"}
TIPO_PARA_NUM = {"lisa": 1.0, "listrada": -1.0, "preta": 2.0}

LARGURA_JANELA_CLIQUE = 1280  # px, só pra exibição na hora de clicar as quinas
ALTURA_JANELA_CLIQUE  = 800   # px — importante pra fotos em retrato (mais altas que largas)


# 1. Carregar/converter a foto crua
def carregar_imagem(caminho):
    """Abre qualquer formato (HEIC, PNG, JPG...) e devolve um array BGR (OpenCV)."""
    img = Image.open(caminho)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# 2. Homografia — clique interativo das 4 quinas
def coletar_quinas(img_bgr):
    """Abre uma janela OpenCV e espera 4 cliques: TL -> TR -> BR -> BL."""
    h, w = img_bgr.shape[:2]
    escala = min(1.0, LARGURA_JANELA_CLIQUE / w, ALTURA_JANELA_CLIQUE / h)
    disp = cv2.resize(img_bgr, (int(w * escala), int(h * escala)))

    labels = ["TL", "TR", "BR", "BL"]
    cores  = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 165, 255)]
    pontos = []

    janela = "Clique nas 4 quinas da mesa: TL -> TR -> BR -> BL  (ESC cancela)"
    cv2.namedWindow(janela)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pontos) < 4:
            pontos.append((x / escala, y / escala))
            cv2.circle(disp, (x, y), 8, cores[len(pontos) - 1], -1)
            cv2.putText(disp, labels[len(pontos) - 1], (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cores[len(pontos) - 1], 2)
            cv2.imshow(janela, disp)

    cv2.setMouseCallback(janela, on_click)
    cv2.imshow(janela, disp)

    print("Clique nas 4 quinas da mesa, na ordem: TL (superior-esq) -> TR -> BR -> BL")
    while len(pontos) < 4:
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == 27:  # ESC
            cv2.destroyWindow(janela)
            print("Cancelado pelo usuário.")
            sys.exit(1)
    cv2.waitKey(300)
    cv2.destroyWindow(janela)
    return pontos


def aplicar_homografia(img_bgr, pontos):
    pts_src = np.float32(pontos)
    pts_dst = np.float32([
        [0, 0],
        [LARGURA_MESA, 0],
        [LARGURA_MESA, ALTURA_MESA],
        [0, ALTURA_MESA],
    ])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img_bgr, M, (LARGURA_MESA, ALTURA_MESA))


# 3. Detecção de bolas (YOLO)
def detectar_bolas_e_cacapas(imagem, modelo, conf=0.45, iou=0.35):
    altura, largura = imagem.shape[:2]
    resultados = {"cacapas": [], "bolas": []}

    margem_x = int(largura * 0.02)
    margem_y = int(altura * 0.04)
    resultados["cacapas"] = [
        {"posicao": "superior_esq",  "x": margem_x,            "y": margem_y},
        {"posicao": "superior_meio", "x": largura // 2,        "y": margem_y},
        {"posicao": "superior_dir",  "x": largura - margem_x,  "y": margem_y},
        {"posicao": "inferior_esq",  "x": margem_x,            "y": altura - margem_y},
        {"posicao": "inferior_meio", "x": largura // 2,        "y": altura - margem_y},
        {"posicao": "inferior_dir",  "x": largura - margem_x,  "y": altura - margem_y},
    ]

    pred = modelo.predict(imagem, conf=conf, iou=iou, verbose=False)[0]
    for box in pred.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(largura, x2), min(altura, y2)

        classe_id = int(box.cls[0])
        tipo      = NOMES_CLASSES.get(classe_id, "lisa")
        cx, cy    = (x1 + x2) // 2, (y1 + y2) // 2
        raio      = max((x2 - x1), (y2 - y1)) // 2

        resultados["bolas"].append({
            "tipo": tipo, "x": cx, "y": cy, "raio": raio,
            "confianca": round(float(box.conf[0]), 3),
        })
    return resultados


# 4. Regras do 8-ball + montagem do vetor de features
def filtrar_candidatas(bolas, meu_time):
    sem_branca     = [b for b in bolas if b["tipo"] != "branca"]
    tem_bola_minha = any(b["tipo"] == meu_time for b in sem_branca)
    validas = []
    for b in sem_branca:
        if b["tipo"] == meu_time:
            validas.append(b)
        elif b["tipo"] == "preta" and not tem_bola_minha:
            validas.append(b)
    return validas


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _ang(a, b):
    return math.atan2(b["y"] - a["y"], b["x"] - a["x"])


def montar_vetor(bolas, branca, feature_cols, cacapas):
    outras = [b for b in bolas if b["tipo"] != "branca"]
    outras.sort(key=lambda b: _dist(branca, b))

    vetor = {
        "branca_x": branca["x"] / LARGURA_MESA,
        "branca_y": branca["y"] / ALTURA_MESA,
    }
    for i, bola in enumerate(outras[:15], 1):
        d_br = _dist(branca, bola)
        a_br = _ang(branca, bola)

        dists_cacapas = [_dist(bola, c) for c in cacapas]
        idx_prox   = dists_cacapas.index(min(dists_cacapas))
        ang_prox   = _ang(bola, cacapas[idx_prox])
        diff_corte = (ang_prox - a_br + math.pi) % (2 * math.pi) - math.pi

        vetor[f"b{i}_tipo"]        = TIPO_PARA_NUM.get(bola["tipo"], 0.0)
        vetor[f"b{i}_x"]           = bola["x"] / LARGURA_MESA
        vetor[f"b{i}_y"]           = bola["y"] / ALTURA_MESA
        vetor[f"b{i}_dist_branca"] = d_br / LARGURA_MESA
        vetor[f"b{i}_ang_branca"]  = math.sin(a_br)
        vetor[f"b{i}_dist_cacapa"] = dists_cacapas[idx_prox] / math.hypot(LARGURA_MESA, ALTURA_MESA)
        vetor[f"b{i}_ang_corte"]   = abs(diff_corte) / math.pi

    for i in range(len(outras) + 1, 16):
        for campo in ["tipo", "x", "y", "dist_branca", "ang_branca", "dist_cacapa", "ang_corte"]:
            vetor[f"b{i}_{campo}"] = 0.0

    df = pd.DataFrame([vetor]).reindex(columns=feature_cols, fill_value=0.0)
    return df, outras


# 5. Visualização final
def salvar_visualizacao(img_norm, branca, candidatas, todas_sem_branca, bola_sugerida,
                         meu_time, confianca, caminho_saida):
    vis = img_norm.copy()

    for bola in todas_sem_branca:
        cv2.line(vis, (branca["x"], branca["y"]), (bola["x"], bola["y"]), (100, 100, 100), 1)

    for bola in candidatas:
        cor = (0, 200, 255) if bola["tipo"] == meu_time else (200, 200, 0)
        cv2.line(vis, (branca["x"], branca["y"]), (bola["x"], bola["y"]), cor, 2)
        cv2.circle(vis, (bola["x"], bola["y"]), bola.get("raio", 15) + 4, cor, 2)

    if bola_sugerida is not None:
        cv2.arrowedLine(vis, (branca["x"], branca["y"]),
                         (bola_sugerida["x"], bola_sugerida["y"]),
                         (0, 255, 0), 3, tipLength=0.05)
        cv2.circle(vis, (bola_sugerida["x"], bola_sugerida["y"]),
                   bola_sugerida.get("raio", 15) + 8, (0, 255, 0), 3)
        texto = f"{bola_sugerida['tipo']}  {confianca * 100:.0f}%"
        cv2.putText(vis, texto,
                    (bola_sugerida["x"] - 30, bola_sugerida["y"] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(caminho_saida, vis)
    print(f"Visualização salva em: {caminho_saida}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SinucaVision — sugere a melhor jogada a partir de uma foto crua da mesa.")
    parser.add_argument("foto", help="Caminho da foto crua da mesa (qualquer formato: JPG, PNG, HEIC...)")
    parser.add_argument("--time", choices=["lisa", "listrada"], required=True, help="Seu grupo de bolas")
    parser.add_argument("--saida", default=None, help="Caminho da imagem de saída com a visualização (padrão: data/inferir-jogada-results/<foto>_sugestao.png)")
    args = parser.parse_args()

    if not os.path.exists(CAMINHO_MODELO_YOLO):
        sys.exit(f"[ERRO] Modelo YOLO não encontrado em {CAMINHO_MODELO_YOLO}")
    if not os.path.exists(CAMINHO_MODELO_RF):
        sys.exit(f"[ERRO] Modelo Random Forest não encontrado em {CAMINHO_MODELO_RF}. "
                  f"Rode o modulo3/07_treino_rf.ipynb primeiro.")

    print(f"[1/5] Carregando foto: {args.foto}")
    img_bruta = carregar_imagem(args.foto)

    print("[2/5] Homografia — normalizando a perspectiva da mesa...")
    quinas = coletar_quinas(img_bruta)
    img_norm = aplicar_homografia(img_bruta, quinas)

    print("[3/5] Detectando bolas com o modelo YOLO...")
    modelo_yolo = YOLO(CAMINHO_MODELO_YOLO)
    dados = detectar_bolas_e_cacapas(img_norm, modelo_yolo)
    bolas = dados["bolas"]
    cacapas = dados["cacapas"]
    branca = next((b for b in bolas if b["tipo"] == "branca"), None)

    print(f"      {len(bolas)} bolas detectadas.")
    if branca is None:
        sys.exit("[ERRO] Bola branca não detectada nesta foto — não é possível sugerir jogada. "
                  "Tente novamente com melhor iluminação/ângulo.")

    print("[4/5] Aplicando regras do 8-ball e o Random Forest...")
    candidatas = filtrar_candidatas(bolas, args.time)
    if not candidatas:
        sys.exit("[AVISO] Nenhuma jogada válida encontrada — todas as bolas do seu time já foram encaçapadas "
                  "(ou a preta ainda não está liberada).")

    rf_model = joblib.load(CAMINHO_MODELO_RF)
    feature_cols = rf_model.feature_names_in_.tolist()
    df_inf, todas_sem_branca = montar_vetor(bolas, branca, feature_cols, cacapas)

    idx_sugerido = rf_model.predict(df_inf)[0]
    confianca    = rf_model.predict_proba(df_inf)[0].max()

    if 1 <= idx_sugerido <= len(todas_sem_branca):
        bola_sugerida = todas_sem_branca[idx_sugerido - 1]
    else:
        bola_sugerida = candidatas[0]

    ids_validas = {id(b) for b in candidatas}
    if id(bola_sugerida) not in ids_validas:
        bola_sugerida = candidatas[0]

    print("[5/5] Resultado:")
    print(f"\n>>> Jogada sugerida: bola {bola_sugerida['tipo']} em ({bola_sugerida['x']}, {bola_sugerida['y']})")
    print(f">>> Confiança: {confianca * 100:.1f}%\n")

    caminho_saida = args.saida
    if caminho_saida is None:
        os.makedirs(CAMINHO_RESULTADOS, exist_ok=True)
        nome_base = os.path.splitext(os.path.basename(args.foto))[0]
        caminho_saida = os.path.join(CAMINHO_RESULTADOS, f"{nome_base}_sugestao.png")

    salvar_visualizacao(img_norm, branca, candidatas, todas_sem_branca,
                        bola_sugerida, args.time, confianca, caminho_saida)


if __name__ == "__main__":
    main()
