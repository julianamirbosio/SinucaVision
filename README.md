# SinucaAgent

### estrutura projeto assistente inteligente de sinuca:
- repositório github com 3 pastas contendo os 3 módulos

#### módulo 1: visão computacional (processamento) - experimentos no google colab, código final em .py
- entrada: imagens (sintéticas ou reais)
- saída: vetor/tabela de dados

#### módulo 2: extração de engenharia/física - código .py
- entrada: vetor/tabela de dados
- saída: dados numéricos para aprendizado -> arquivo de dados (.csv) salvando diretamente na pasta data (google drive ou github)

#### módulo 3:  IA (random forest) - google colab
- entrada: dados numéricos para aprendizado de máquina -> pandas para análise dados (IA lerá o arquivo .csv e carregará na memória da GPU para treinamento do algoritmo random forest)
- saída: previsão da melhor caçapa (classe) ou ângulo exato (regressão) -> extração para arquivo binário leve (.pkl - biblioteca joblib ou pickle), salvamento na pasta modulo_3

#### script final do projeto (main) 
- importar funções do modulo_1, modulo_2, modulo_3 e carregar arquivo .pkl
- após a função do modulo_3 informar a melhor jogada: bola, caçapa e ângulo, uso de opencv em cima da imagem original para levar as informações obtidas diretamente acima da imagem, como abaixo (possibilidade de virar um modulo_4):

<img width="2816" height="1536" alt="Gemini_Generated_Image_eu3uj1eu3uj1eu3u" src="https://github.com/user-attachments/assets/55950603-7c09-43da-8f25-135d5bdd0021" />

