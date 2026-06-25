"""
augment_dataset.py
-------------------
Expande um dataset YOLO pequeno (ex: 5 imagens anotadas) em N variações
sintéticas, aplicando transformações geométricas e fotométricas que
simulam a variação de iluminação de bar/salão (sombras, reflexos,
brilho/contraste, ruído de câmera).

ENTRADA esperada (formato YOLO):
  dataset_origem/
    images/   IMG_0640.jpg, IMG_0641.jpg, ...
    labels/   IMG_0640.txt, IMG_0641.txt, ...   (uma linha por bola: "0 cx cy w h" normalizado 0-1)

SAÍDA:
  dataset_aumentado/
    images/train/  *.jpg
    labels/train/  *.txt
    images/val/    *.jpg
    labels/val/    *.txt

Uso:
  pip install albumentations opencv-python-headless --break-system-packages
  python augment_dataset.py --src dataset_origem --dst dataset_aumentado --n 30
  (--n = número de variações geradas POR imagem original)
"""

import os
import argparse
import random
import glob
import cv2
import numpy as np
import albumentations as A


def carregar_labels_yolo(caminho_txt):
    """Lê arquivo YOLO -> lista de [classe, cx, cy, w, h] normalizados."""
    boxes, classes = [], []
    if not os.path.exists(caminho_txt):
        return boxes, classes
    with open(caminho_txt) as f:
        for linha in f:
            partes = linha.strip().split()
            if len(partes) != 5:
                continue
            c, cx, cy, w, h = partes
            classes.append(int(c))
            boxes.append([float(cx), float(cy), float(w), float(h)])
    return boxes, classes


def salvar_labels_yolo(caminho_txt, boxes, classes):
    with open(caminho_txt, 'w') as f:
        for (cx, cy, w, h), c in zip(boxes, classes):
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def adicionar_sombra_sintetica(img, intensidade_max=0.5):
    """Desenha um polígono semi-transparente escuro sobre a imagem,
    simulando a sombra de um jogador ou objeto sobre a mesa."""
    h, w = img.shape[:2]
    overlay = img.copy()
    n_pontos = random.randint(3, 6)
    pontos = np.array([
        [random.randint(0, w), random.randint(0, h)] for _ in range(n_pontos)
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pontos], (0, 0, 0))
    alpha = random.uniform(0.15, intensidade_max)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def adicionar_reflexo_sintetico(img, intensidade_max=0.4):
    """Mancha clara (reflexo de luminária) — simula brilho direto sobre bolas lisas."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cx, cy = random.randint(0, w), random.randint(0, h)
    raio = random.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.2))
    cv2.circle(overlay, (cx, cy), raio, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
    alpha = random.uniform(0.1, intensidade_max)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# Pipeline geométrico + fotométrico (bboxes seguem automaticamente)
def construir_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.9),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.7),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.CLAHE(p=0.2),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['classes']))


def gerar_variacoes(caminho_img, caminho_txt, pasta_img_out, pasta_lbl_out, prefixo, n):
    img = cv2.imread(caminho_img)
    if img is None:
        print(f'  [ERRO] não foi possível ler {caminho_img}')
        return 0

    boxes, classes = carregar_labels_yolo(caminho_txt)
    if not boxes:
        print(f'  [AVISO] {caminho_txt} sem anotações — pulando')
        return 0

    transform = construir_transform()
    gerados = 0

    for i in range(n):
        try:
            aug = transform(image=img, bboxes=boxes, classes=classes)
        except Exception as e:
            # bbox pode sair da imagem após rotação/flip em casos raros; pula essa tentativa
            continue

        img_aug = aug['image']
        boxes_aug = aug['bboxes']
        classes_aug = aug['classes']

        if not boxes_aug:
            continue  # transformação removeu todas as bolas (raro, mas possível com rotação)

        # Augmentation extra "manual" simulando ambiente de bar (não afeta bboxes)
        if random.random() < 0.5:
            img_aug = adicionar_sombra_sintetica(img_aug)
        if random.random() < 0.3:
            img_aug = adicionar_reflexo_sintetico(img_aug)

        nome_saida = f'{prefixo}_aug{i:03d}'
        cv2.imwrite(os.path.join(pasta_img_out, nome_saida + '.jpg'), img_aug)
        salvar_labels_yolo(os.path.join(pasta_lbl_out, nome_saida + '.txt'), boxes_aug, classes_aug)
        gerados += 1

    return gerados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Pasta com images/ e labels/ originais')
    ap.add_argument('--dst', required=True, help='Pasta de saída do dataset aumentado')
    ap.add_argument('--n', type=int, default=30, help='Variações geradas por imagem original')
    ap.add_argument('--val_split', type=float, default=0.15, help='Fração para validação')
    args = ap.parse_args()

    imagens_src = sorted(
        glob.glob(os.path.join(args.src, 'images', '*.jpg')) +
        glob.glob(os.path.join(args.src, 'images', '*.JPG')) +
        glob.glob(os.path.join(args.src, 'images', '*.png'))
    )
    print(f'Imagens originais encontradas: {len(imagens_src)}')
    if not imagens_src:
        print('Nenhuma imagem encontrada em', os.path.join(args.src, 'images'))
        return

    # Cria estrutura final train/val
    for split in ['train', 'val']:
        os.makedirs(os.path.join(args.dst, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, 'labels', split), exist_ok=True)

    # Split determinístico: garante pelo menos 1 imagem original em val,
    # mesmo com poucas imagens (random puro pode jogar todas pro mesmo lado).
    n_val = max(1, round(len(imagens_src) * args.val_split))
    indices_val = set(random.sample(range(len(imagens_src)), k=min(n_val, len(imagens_src))))
    if len(imagens_src) > 1 and len(indices_val) == len(imagens_src):
        # nunca deixa train vazio se houver mais de 1 imagem
        indices_val = set(list(indices_val)[:-1])
    print(f'Split: {len(imagens_src) - len(indices_val)} imagens em train, {len(indices_val)} em val')

    total_gerado = 0
    for idx, caminho_img in enumerate(imagens_src):
        nome_base = os.path.splitext(os.path.basename(caminho_img))[0]
        caminho_txt = os.path.join(args.src, 'labels', nome_base + '.txt')

        split = 'val' if idx in indices_val else 'train'
        pasta_img_out = os.path.join(args.dst, 'images', split)
        pasta_lbl_out = os.path.join(args.dst, 'labels', split)

        # Copia também o original (sem augmentation) para o dataset final
        img_original = cv2.imread(caminho_img)
        cv2.imwrite(os.path.join(pasta_img_out, nome_base + '_orig.jpg'), img_original)
        boxes, classes = carregar_labels_yolo(caminho_txt)
        salvar_labels_yolo(os.path.join(pasta_lbl_out, nome_base + '_orig.txt'), boxes, classes)

        gerados = gerar_variacoes(caminho_img, caminho_txt, pasta_img_out, pasta_lbl_out,
                                   prefixo=nome_base, n=args.n)
        total_gerado += gerados
        print(f'  {nome_base}: {gerados} variações geradas → {split}/')

    print(f'\nTotal de imagens no dataset aumentado: {total_gerado + len(imagens_src)}')
    print(f'Saída em: {args.dst}')

    # Gera o data.yaml para o treino do YOLO
    yaml_path = os.path.join(args.dst, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(os.path.join(args.dst, 'images', 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(args.dst, 'images', 'val'))}\n")
        f.write("nc: 1\n")
        f.write("names: ['bola']\n")
    print(f'data.yaml criado em: {yaml_path}')


if __name__ == '__main__':
    main()
