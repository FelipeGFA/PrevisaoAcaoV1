import os, numpy as np, sys
from tensorflow.keras.models import load_model
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QFileDialog, QWidget, QMessageBox, QGridLayout, 
                           QScrollArea, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QCoreApplication

def carregar_classes():
    with open(os.path.join('ModeloeData', 'data.txt'), 'r') as arquivo:
        return [linha.strip() for linha in arquivo.readlines()]

def carregar_modelo():
    try: return load_model(os.path.join('ModeloeData', 'modelo_treinado.keras'))
    except Exception: return None

def preprocessar_imagem(caminho_imagem, tamanho=(160, 160)):
    try:
        img = cv2.imread(caminho_imagem)
        if img is None: return None
        img = cv2.resize(img, tamanho)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        return np.expand_dims(img.astype('float32'), axis=0)
    except Exception: return None

def prever_acao(modelo, imagem_processada, classes):
    if modelo is None or imagem_processada is None: return "Erro", 0, None
    try:
        predicoes = modelo.predict(imagem_processada)
        indice_classe = np.argmax(predicoes[0])
        return classes[indice_classe], float(predicoes[0][indice_classe]), predicoes[0]
    except Exception: return "Erro", 0, None

def adicionar_texto_imagem(imagem, texto_principal):
    img_com_texto = imagem.copy()
    altura, largura = img_com_texto.shape[:2]
    espessura = max(1, int(largura / 400))
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_fonte = max(0.7, largura / 500)
    (w_texto, h_texto), _ = cv2.getTextSize(texto_principal, fonte, escala_fonte, espessura)
    y_fundo_fim = h_texto + 40
    pos_texto = ((largura - w_texto) // 2, h_texto + 20)
    overlay = img_com_texto.copy()
    cv2.rectangle(overlay, (0, 0), (largura, y_fundo_fim), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img_com_texto, 0.4, 0, img_com_texto)
    cv2.putText(img_com_texto, texto_principal, pos_texto, fonte, escala_fonte, (0, 255, 0), espessura, cv2.LINE_AA)
    return img_com_texto

def converter_cv_para_qt(cv_img):
    if cv_img is None: return QPixmap()
    if cv_img.dtype == np.float32 or cv_img.dtype == np.float64:
        cv_img = (cv_img * 255).astype(np.uint8)
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    altura, largura = cv_img_rgb.shape[:2]
    q_img = QImage(cv_img_rgb.data, largura, altura, 3 * largura, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)

class CaixaImagemResultado(QFrame):
    def __init__(self, caminho_inicial=None, parent=None):
        super().__init__(parent)
        self.caminho = caminho_inicial
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        self.label_imagem = QLabel()
        self.label_imagem.setAlignment(Qt.AlignCenter)
        self.label_imagem.setMinimumSize(200,150)
        self.label_imagem.setScaledContents(True)
        self.label_resultado_texto = QLabel()
        self.label_resultado_texto.setAlignment(Qt.AlignCenter)
        self.label_resultado_texto.setWordWrap(True)
        layout.addWidget(self.label_imagem)
        layout.addWidget(self.label_resultado_texto)
        if caminho_inicial: self.carregar_imagem_inicial(caminho_inicial)

    def carregar_imagem_inicial(self, caminho_img):
        self.caminho = caminho_img
        try:
            img_cv = cv2.imread(caminho_img)
            if img_cv is None: raise ValueError()
            self.label_imagem.setPixmap(converter_cv_para_qt(img_cv))
            self.label_resultado_texto.setText(os.path.basename(caminho_img))
        except Exception:
            self.label_imagem.setText("Erro ao carregar")
            self.label_resultado_texto.setText(f"{os.path.basename(caminho_img)}\n<Erro>")

    def definir_resultado_e_imagem_anotada(self, classe, probabilidade, imagem_anotada_cv=None):
        self.label_resultado_texto.setText(f"{classe} ({probabilidade:.2%})")
        if imagem_anotada_cv is not None:
            self.label_imagem.setPixmap(converter_cv_para_qt(imagem_anotada_cv))

class InterfacePrevisao(QMainWindow):
    def __init__(self):
        super().__init__()
        self.caminhos_imagens = []
        self.widgets_caixas_resultado = {}
        self.classes = carregar_classes()
        self.modelo = carregar_modelo()
        self._inicializar_ui()

    def _inicializar_ui(self):
        self.setWindowTitle("Previsão de Ações")
        self.setGeometry(100, 100, 900, 700)
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)
        
        painel_botoes_superior = QHBoxLayout()
        painel_botoes_superior.setAlignment(Qt.AlignCenter)
        
        self.btn_selecionar = QPushButton("Selecionar Imagens")
        self.btn_selecionar.clicked.connect(self._selecionar_imagens)
        self.btn_selecionar.setMinimumHeight(40)
        self.btn_selecionar.setMinimumWidth(150)
        self.btn_selecionar.setFont(QFont('Arial', 10))
        
        self.btn_prever = QPushButton("Prever")
        self.btn_prever.clicked.connect(self._prever_imagens_carregadas)
        self.btn_prever.setEnabled(False)
        self.btn_prever.setMinimumHeight(40)
        self.btn_prever.setMinimumWidth(150)
        self.btn_prever.setFont(QFont('Arial', 10))
        
        self.btn_limpar = QPushButton("Limpar")
        self.btn_limpar.clicked.connect(self._limpar_tudo)
        self.btn_limpar.setEnabled(False)
        self.btn_limpar.setMinimumHeight(40)
        self.btn_limpar.setMinimumWidth(150)
        self.btn_limpar.setFont(QFont('Arial', 10))
        
        painel_botoes_superior.addStretch(1)
        painel_botoes_superior.addWidget(self.btn_selecionar)
        painel_botoes_superior.addSpacing(20)
        painel_botoes_superior.addWidget(self.btn_prever)
        painel_botoes_superior.addSpacing(20)
        painel_botoes_superior.addWidget(self.btn_limpar)
        painel_botoes_superior.addStretch(1)
        
        layout_principal.addLayout(painel_botoes_superior)
        
        scroll_area_resultados = QScrollArea()
        scroll_area_resultados.setWidgetResizable(True)
        layout_principal.addWidget(scroll_area_resultados)
        self.container_grid_resultados = QWidget()
        self.grid_layout_resultados = QGridLayout(self.container_grid_resultados)
        self.grid_layout_resultados.setSpacing(10)
        scroll_area_resultados.setWidget(self.container_grid_resultados)
        
        if self.modelo is None: QMessageBox.critical(self, "Erro", "Não foi possível carregar o modelo.")

    def _selecionar_imagens(self):
        caminhos_selecionados = QFileDialog.getOpenFileNames(self, "Selecionar Imagens", "", 
                                            "Imagens (*.png *.jpg *.jpeg *.bmp);;Todos os arquivos (*)", 
                                            options=QFileDialog.Options())[0]
        if caminhos_selecionados:
            self.caminhos_imagens = caminhos_selecionados
            self._atualizar_grid_com_imagens(self.caminhos_imagens)
            self.btn_prever.setEnabled(True)
            self.btn_limpar.setEnabled(True)

    def _limpar_grid_widgets(self):
        while self.grid_layout_resultados.count():
            item_widget = self.grid_layout_resultados.takeAt(0).widget()
            if item_widget: item_widget.deleteLater()
        self.widgets_caixas_resultado.clear()

    def _atualizar_grid_com_imagens(self, lista_caminhos_img):
        self._limpar_grid_widgets()
        num_colunas = max(1, self.container_grid_resultados.width() // 220)
        for idx, caminho in enumerate(lista_caminhos_img):
            caixa = CaixaImagemResultado(caminho)
            self.grid_layout_resultados.addWidget(caixa, idx // num_colunas, idx % num_colunas)
            self.widgets_caixas_resultado[caminho] = caixa
    
    def _limpar_tudo(self):
        self._limpar_grid_widgets()
        self.caminhos_imagens = []
        self.btn_prever.setEnabled(False)
        self.btn_limpar.setEnabled(False)

    def _processar_e_anotar_imagem_cv(self, caminho_img):
        imagem_para_modelo = preprocessar_imagem(caminho_img)
        if imagem_para_modelo is None: return "Erro", 0, None
        
        classe, prob, _ = prever_acao(self.modelo, imagem_para_modelo, self.classes)
        if classe == "Erro": return "Erro", prob, None

        imagem_original_cv = cv2.imread(caminho_img)
        if imagem_original_cv is None: return classe, prob, None
        
        return classe, prob, adicionar_texto_imagem(imagem_original_cv, f"{classe} ({prob:.2%})")

    def _prever_imagens_carregadas(self):
        if not self.caminhos_imagens: return
        
        for caminho_img_atual in self.caminhos_imagens:
            classe_pred, prob_pred, img_anotada_cv = self._processar_e_anotar_imagem_cv(caminho_img_atual)
            caixa_widget = self.widgets_caixas_resultado.get(caminho_img_atual)
            if caixa_widget:
                caixa_widget.definir_resultado_e_imagem_anotada(classe_pred, prob_pred, img_anotada_cv)
            QCoreApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    janela = InterfacePrevisao()
    janela.show()
    sys.exit(app.exec_()) 