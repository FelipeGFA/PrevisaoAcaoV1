# Previsão de Ações Humanas 1.0

Uma aplicação simples para classificação e predição de ações humanas em imagens utilizando modelos de deep learning.

**Aviso: Essa é uma versão protótipo, os resultados apresentam pouca precisão.**

## Funcionalidades

- Interface gráfica com PyQt5
- Carregamento e processamento de imagens
- Previsão e classificação de ações humanas
- Visualização de resultados com porcentagem de confiança
- Processamento em lote de múltiplas imagens

## Requisitos

- Python 3.6+
- TensorFlow 2.x
- OpenCV (cv2)
- PyQt5
- NumPy

## Instalação

1. Clone o repositório:
```
git clone https://github.com/FelipeGFA/PrevisaoAcaoV1.git
cd PrevisaoAcao
```

2. Configure um ambiente virtual e instale as dependências:
```
python -m venv venv
venv\scripts\activate
pip install tensorflow opencv-python pyqt5 numpy
```

## Como usar

3. Execute o aplicativo:
```
python previsao.py
```

4. Na interface:
   - Clique em "Selecionar Imagens" para escolher arquivos de imagem
   - Clique em "Prever" para processar e classificar as imagens
   - Clique em "Limpar" para remover as imagens da interface

## Estrutura do Projeto

- `previsao.py`: Código principal com interface e lógica de previsão
- `ModeloeData/`: Diretório com modelo treinado e dados das classes
  - `modelo_treinado.keras`: Modelo de classificação de ações
  - `data.txt`: Lista de classes de ações humanas

