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
│  Módulo 1   │  Visão clássica — OpenCV
│             │  Converte, normaliza perspectiva e detecta bolas
└──────┬──────┘
       │ coordenadas (X, Y) de cada bola
       ▼
┌─────────────┐
│  Módulo 2   │  Engenharia de features — NumPy
│             │  Calcula distâncias, ângulos e gera vetor fixo
└──────┬──────┘
       │ vetor numérico normalizado
       ▼
┌─────────────┐
│  Módulo 3   │  IA — scikit-learn
│             │  Random Forest prevê a melhor caçapa e ângulo
└─────────────┘
```

---

## Estrutura de pastas

```
SINUCAVISION/
├── modulo1/
│   ├── 00_converter_heic.ipynb   # converte qualquer imagem para JPG
│   └── 01_homografia.ipynb       # normaliza perspectiva (top-down)
├── modulo2/
│   └── 04_feature_eng.ipynb      # gera dataset.csv com features geométricas
├── modulo3/
│   ├── 05_treino_rf.ipynb        # treina o modelo Random Forest
│   └── 06_inferencia.ipynb       # faz previsões em novas fotos
├── data/                         # ignorada pelo git (ver .gitignore)
│   ├── raw/                      # fotos originais (HEIC, PNG, etc.)
│   ├── converted/                # fotos convertidas para JPG
│   ├── normalized/               # fotos em top-down 800×400 px
│   ├── annotations/              # quinas da mesa (JSON por imagem)
│   └── dataset.csv               # features extraídas (gerado pelo módulo 2)
├── models/                       # modelos treinados (.pkl) — ignorado pelo git
├── utils.py                      # constantes e funções compartilhadas
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pré-requisitos

- Python 3.10+
- pip
- Git

---

## Como rodar localmente (VS Code ou JupyterLab)

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/SINUCAVISION.git
cd SINUCAVISION
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

Cada notebook já inclui a célula de montagem do Drive.
Basta abrir o notebook no Colab e executar célula por célula.

O dataset fica em:
```
Google Drive > Colab Notebooks > SinucaVision > data/
```

---

## Ordem de execução dos notebooks

| Passo | Notebook | O que faz |
|-------|----------|-----------|
| 1 | `modulo1/00_converter_heic.ipynb` | Converte fotos de `raw/` para JPG em `converted/` |
| 2 | `modulo1/01_homografia.ipynb` | Clique nas 4 quinas → gera `normalized/` (top-down) |
| 3 | `modulo1/02_detector_bolas.ipynb` | Detecta bolas com HoughCircles + HSV, salva JSON |
| 4 | `modulo1/03_visualizador.ipynb` | Visualiza detecções para debug |
| 5 | `modulo2/04_feature_eng.ipynb` | Gera `data/dataset.csv` com vetor de features |
| 6 | `modulo3/05_treino_rf.ipynb` | Treina Random Forest, salva `models/modelo.pkl` |
| 7 | `modulo3/06_inferencia.ipynb` | Carrega modelo e faz previsões em novas fotos |

---

## Dataset

As fotos do dataset foram tiradas com iPhone (formato HEIC).
O notebook `00_converter_heic` suporta:

| Extensão | Suporte |
|----------|---------|
| `.HEIC` / `.heic` | ✅ via pillow-heif |
| `.JPG` / `.JPEG` | ✅ nativo |
| `.PNG` | ✅ nativo |
| `.WEBP` | ✅ nativo |
| `.TIFF` / `.TIF` | ✅ nativo |
| `.BMP` | ✅ nativo |

As fotos **não são commitadas** no repositório (`.gitignore`).

---

## Vetor de features (Módulo 2)

Cada foto gera uma linha no `dataset.csv`.
As colunas seguem sempre a mesma ordem, independente de quantas bolas estão na mesa:

```
X_branca, Y_branca,
X_amarela, Y_amarela, dist_branca_amarela, ang_branca_amarela, presente_amarela,
X_azul,    Y_azul,    dist_branca_azul,    ang_branca_azul,    presente_azul,
... (uma linha por bola)
→ label: qual caçapa (0–5)
```

Bolas ausentes recebem `-1` em todas as suas colunas.

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
ipykernel
```

Veja o arquivo `requirements.txt` para versões exatas.

<img width="2816" height="1536" alt="Gemini_Generated_Image_eu3uj1eu3uj1eu3u" src="https://github.com/user-attachments/assets/55950603-7c09-43da-8f25-135d5bdd0021" />

