# SinucaVision 🎱

Agente de sinuca baseado em visão computacional e aprendizado de máquina.
O sistema identifica a posição das bolas na mesa e recomenda a melhor jogada.

Trabalho desenvolvido para a disciplina de Visão Computacional.

---

## Arquitetura

```
Foto da mesa
     │
     ▼
┌─────────────┐
│  Módulo 1   │  OpenCV + YOLOv8
│             │  Converte, normaliza a perspectiva (homografia) e detecta
│             │  as bolas/caçapas com um modelo YOLO treinado
└──────┬──────┘
       │ JSON com tipo e coordenadas (X, Y) de cada bola + caçapas
       ▼
┌─────────────┐
│  Módulo 2   │  Engenharia de features — NumPy
│             │  Calcula distâncias e ângulos, gera vetor fixo de 107 colunas
└──────┬──────┘
       │ vetor numérico normalizado (data/dataset.csv)
       ▼
┌─────────────┐
│  Módulo 3   │  IA — scikit-learn
│             │  Random Forest + regras do 8-ball sugerem a melhor bola/jogada
└─────────────┘
```

---

## Estrutura de pastas

```
SinucaVision/
├── modulo1/
│   ├── 00_converter_heic.ipynb    # converte qualquer imagem para JPG
│   ├── 01_homografia.ipynb        # normaliza perspectiva (top-down) — interativo
│   ├── 02_aumentar_dataset.ipynb  # aumento de dados (augmentation) p/ treinar o YOLO — uso pontual
│   ├── 03_treino_yolo.ipynb       # treina o modelo YOLOv8 de detecção de bolas — uso pontual (Colab/GPU)
│   ├── 04_detector_bolas.ipynb    # detecta bolas/caçapas com o YOLO treinado
│   └── 05_visualizador.ipynb      # visualiza detecções para debug (não salva arquivo)
├── modulo2/
│   └── 06_feature_eng.ipynb       # gera dataset.csv com features geométricas
├── modulo3/
│   ├── 07_treino_rf.ipynb         # treina o Random Forest, salva models/random_forest_jogada.pkl
│   └── 08_inferencia.ipynb        # recomenda jogada para uma imagem já processada
├── inferir_jogada.py              # CLI: foto crua → jogada sugerida, num comando só
├── data/                          # ignorada pelo git (ver .gitignore)
│   ├── raw/                       # fotos originais (HEIC, PNG, etc.)
│   ├── converted/                 # fotos convertidas para JPG
│   ├── normalized/                # fotos em top-down 800×400 px
│   ├── annotations/                # quinas da mesa (JSON por imagem)
│   ├── output/
│   │   └── <nome>_balls.json      # bolas/caçapas detectadas por imagem
│   └── dataset.csv                # features extraídas (gerado pelo módulo 2)
├── models/                        # ignorado pelo git — ver "Como criar os modelos"
│   ├── yolo_deteccao_bolas.pt     # detector de bolas (YOLOv8)
│   └── random_forest_jogada.pkl   # Random Forest de sugestão de jogada
├── rodar_pipeline.sh              # roda o pipeline de detecção → features → treino RF
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pré-requisitos

- Python 3.10+
- pip
- Git

> A detecção de bolas usa YOLOv8 (`ultralytics`, que traz `torch` junto). A instalação é pesada (a versão CPU é a mais leve e suficiente pra rodar o modelo já treinado — só o **treino** do YOLO se beneficia de GPU/Colab).

---

## Como rodar localmente (VS Code ou JupyterLab)

### 1. Clone o repositório

```bash
git clone git@github.com:julianamirbosio/SinucaVision.git
cd SinucaVision
```

### 2. Crie e ative o ambiente virtual

```bash
python3 -m venv .venv

# Linux / Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

> Se a instalação do `ultralytics`/`torch` travar ou cair por timeout de rede, instale o `torch` primeiro na versão CPU (bem mais leve) e depois o resto:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4. Registre o kernel no Jupyter

```bash
python -m ipykernel install --user --name=sinucavision --display-name "SinucaVision"
```

### 5. Abra o JupyterLab

```bash
jupyter lab
```

> Selecione o kernel **SinucaVision** no canto superior direito de cada notebook.

---

## Como rodar no Google Colab

Os notebooks de treino (`00`–`03`) incluem célula de montagem do Drive e são os mais pesados para rodar localmente — recomendado usá-los no Colab com GPU.
Basta abrir o notebook no Colab e executar célula por célula.

O dataset fica em:
```
Google Drive > Colab Notebooks > SinucaVision > data/
```

---

## Como criar os modelos

O projeto depende de **dois modelos treinados**, ambos em `models/`..

### 1. `yolo_deteccao_bolas.pt` — detector de bolas (YOLOv8)

Treinado a partir de um dataset anotado manualmente no [Roboflow](https://roboflow.com) (fotos da mesa com as caixas de cada bola marcadas).

1. Exporte o dataset anotado do Roboflow.
2. `modulo1/02_aumentar_dataset.ipynb` — gera variações sintéticas (rotação, espelhamento, brilho, ruído) a partir de `data/dataset_origem/`, salva em `data/dataset_aumentado/`.
3. `modulo1/03_treino_yolo.ipynb` — treina o YOLOv8 em cima do dataset aumentado. **Rode no Google Colab com GPU** (compacte `dataset_aumentado/` e faça upload) — treinar localmente em CPU é inviável na prática.
4. Baixe o `best.pt` gerado e salve como `models/yolo_deteccao_bolas.pt`.

Esse passo só precisa ser refeito se quiser melhorar a detecção (mais dados, mais classes, etc.) — no dia a dia, o modelo já treinado é reutilizado.

### 2. `random_forest_jogada.pkl` — Random Forest de sugestão de jogada

Depende do `yolo_deteccao_bolas.pt` já existir em `models/`. Uma vez com fotos processadas:

```bash
./rodar_pipeline.sh
```

Isso roda a detecção de bolas + engenharia de features + treino do RF, e salva `models/random_forest_jogada.pkl` no final (detalhes na seção abaixo). Pode ser refeito a qualquer momento que o dataset crescer — não exige GPU.

---

## Ordem de execução dos notebooks

| Passo | Notebook | O que faz |
|-------|----------|-----------|
| 1 | `modulo1/00_converter_heic.ipynb` | Converte fotos de `raw/` para JPG em `converted/` |
| 2 | `modulo1/01_homografia.ipynb` | Clique nas 4 quinas → gera `normalized/` (top-down) ⚠ interativo |
| 3 | `modulo1/02_aumentar_dataset.ipynb` | Uso pontual — só para (re)treinar o YOLO |
| 4 | `modulo1/03_treino_yolo.ipynb` | Uso pontual — só para (re)treinar o YOLO, recomendado no Colab |
| 5 | `modulo1/04_detector_bolas.ipynb` | Detecta bolas/caçapas com o YOLO treinado, salva JSON por imagem |
| 6 | `modulo1/05_visualizador.ipynb` | Visualiza detecções para debug (uso pontual) |
| 7 | `modulo2/06_feature_eng.ipynb` | Gera `data/dataset.csv` com vetor de features |
| 8 | `modulo3/07_treino_rf.ipynb` | Treina Random Forest, salva `models/random_forest_jogada.pkl` |
| 9 | `modulo3/08_inferencia.ipynb` | Recomenda jogada para uma imagem já processada (uso pontual) |

Os passos 3 e 4 só entram em jogo quando o YOLO precisa ser retreinado (ver "Como criar os modelos"). No uso corrente, o fluxo é: 1 → 2 → 5 → 7 → 8 (ou `./rodar_pipeline.sh` para os passos 5, 7 e 8-equivalente).

---

## Pipeline de treino automatizado (`rodar_pipeline.sh`)

Os passos 1 e 2 da tabela acima exigem execução manual (conversão de arquivos e anotação interativa das quinas). Feito isso — e com `models/yolo_deteccao_bolas.pt` já presente —, os passos seguintes podem ser executados de uma vez com:

```bash
./rodar_pipeline.sh
```

O script executa os notebooks na ordem correta usando o kernel `sinucavision` e atualiza os outputs de cada um. Qualquer alteração feita nos notebooks é refletida automaticamente na próxima execução.

**O que ele faz:**

| Etapa | Notebook executado | Saída gerada |
|-------|--------------------|--------------|
| 1/3 | `04_detector_bolas` | `data/output/<nome>_balls.json` por imagem |
| 2/3 | `06_feature_eng` | `data/dataset.csv` reconstruído do zero |
| 3/3 | `07_treino_rf` | `models/random_forest_jogada.pkl` atualizado |

> Para usar o modelo treinado numa foto específica já processada, abra `08_inferencia.ipynb` manualmente e configure `NOME_IMAGEM` e `MEU_TIME`.

---

## Uso rápido: sugerir uma jogada (`inferir_jogada.py`)

Pra quem só quer o resultado final — sem abrir nenhum notebook —, existe um script único que recebe uma foto crua da mesa e devolve a jogada sugerida:

```bash
python inferir_jogada.py data/raw/IMG_0640.HEIC --time lisa
```

O que ele faz, em um só comando:

1. Abre a foto (qualquer formato — HEIC, PNG, JPG...)
2. Abre uma janela para você clicar as 4 quinas da mesa (TL → TR → BR → BL) e aplica a homografia
3. Detecta as bolas/caçapas com `models/yolo_deteccao_bolas.pt`
4. Monta o vetor de features e aplica as regras do 8-ball + `models/random_forest_jogada.pkl`
5. Imprime no terminal a bola sugerida e a confiança do modelo
6. Salva `<nome>_sugestao.png` com a jogada desenhada sobre a mesa

Requer os dois modelos treinados em `models/` (ver "Como criar os modelos") e uma tela disponível (não roda em servidor sem display, por causa do clique interativo das quinas).

---

## Artefatos e Disponibilidade de Materiais

| Artefato | Link |
|---|---|
| Checkpoint do Modelo (Rede Neural) | [Google Drive](https://drive.google.com/file/d/1X9WumfFjenlukUJXES4fsWeWb-vANtcF/view?usp=sharing) |
| Dataset Original | [Google Drive](https://drive.google.com/drive/folders/1oBScCcczCjd3nfHsyMiJBUzoA8sLjE3W?usp=sharing) |
| Dataset Aumentado | [Google Drive](https://drive.google.com/drive/folders/1G_k3ajHZvzJDsl_hO4Fv-ONI03h_1qor?usp=sharing) |

---

## Vetor de features (Módulo 2)

Cada imagem processada gera uma linha em `dataset.csv` — **107 colunas**, sempre na mesma ordem, independentemente de quantas bolas estão na mesa:

```
branca_x, branca_y,
b1_tipo, b1_x, b1_y, b1_dist_branca, b1_ang_branca, b1_dist_cacapa, b1_ang_corte,
b2_tipo, b2_x, b2_y, b2_dist_branca, b2_ang_branca, b2_dist_cacapa, b2_ang_corte,
... (até b15, ordenadas por distância à branca)
```

- `tipo`: `1.0` lisa · `-1.0` listrada · `2.0` preta · `0.0` posição vazia (zero-padding)
- `dist_*`: distâncias normalizadas pela diagonal/largura da mesa
- `ang_branca`: seno do ângulo branca → bola
- `ang_corte`: dificuldade da tacada — diferença angular entre o ângulo de ataque (branca → bola) e o ângulo ideal até a caçapa mais próxima
- Bolas ausentes (menos de 15 na mesa) recebem `0.0` em todas as colunas

O `07_treino_rf.ipynb` gera automaticamente a coluna `alvo_bola` (label, índice 1–15 da melhor bola) por heurística, já que não há rótulo manual.

---

## Dependências principais

```
opencv-python
numpy
pandas
scikit-learn
matplotlib
pillow-heif
jupyterlab
ipympl
ultralytics
```

Veja o arquivo `requirements.txt` para versões exatas.
