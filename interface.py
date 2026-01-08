from __future__ import annotations
import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QFileDialog, QFrame, QSizePolicy, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen, QMouseEvent
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import cv2 as cv
from numpy.typing import NDArray


# Thread de traitement d'image - empêche le blocage de l'interface utilisateur
class ImageProcessingThread(QThread):
    finished: pyqtSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    progress: pyqtSignal = pyqtSignal(int)
    error: pyqtSignal = pyqtSignal(str)
    
    def __init__(self: ImageProcessingThread, data: NDArray, kernel_size: int, threshold_multiplier: float):
        super().__init__()
        self.data: NDArray = data
        self.kernel_size: int = kernel_size
        self.threshold_multiplier: float = threshold_multiplier
        
    def run(self):
        try:
            self.progress.emit(10)
            
            # Conversion en uint8
            if self.data.ndim == 3:
                data_transposed: NDArray = np.transpose(self.data, (1, 2, 0)) if self.data.shape[0] == 3 else self.data
                image: NDArray = np.zeros_like(data_transposed, dtype='uint8')
                for i in range(data_transposed.shape[2]):
                    channel: NDArray = data_transposed[:, :, i]
                    image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
                data_gray: NDArray = np.mean(data_transposed, axis=2)
            else:
                image: NDArray = ((self.data - self.data.min()) / (self.data.max() - self.data.min()) * 255).astype('uint8')
                data_gray: NDArray = self.data
                
            self.progress.emit(30)
            
            # Détection des étoiles
            mean: float
            median: float
            std: float

            mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)
            daofind: DAOStarFinder = DAOStarFinder(fwhm=3.0, threshold=self.threshold_multiplier * std)
            sources = daofind(data_gray - median)
            
            self.progress.emit(50)
            
            if sources is None or len(sources) == 0:
                self.error.emit("Aucune étoile détectée")
                self.finished.emit(image, np.zeros_like(data_gray, dtype=np.uint8), 
                                 np.zeros_like(data_gray, dtype=np.uint8), image)
                return
            
            # Création du masque
            mask: NDArray = np.zeros(data_gray.shape, dtype=np.uint8)
            for source in sources:
                x, y = int(source['xcentroid']), int(source['ycentroid'])
                radius = int(max(4, min(12, source['sharpness'] * 8 + source['peak'] / 2000)))
                cv.circle(mask, (x, y), radius, 255, -1)
            
            self.progress.emit(70)
            
            # Érosion du masque
            kernel: NDArray = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            mask_eroded: NDArray = cv.erode(mask, kernel, iterations=1)
            
            # Inpainting
            background: NDArray = cv.inpaint(image, mask_eroded, 3, cv.INPAINT_TELEA)
            
            self.progress.emit(85)
            
            # Mélange progressif
            mask_float: NDArray = mask_eroded.astype(np.float32) / 255.0
            mask_blurred: NDArray = cv.GaussianBlur(mask_float, (11, 11), 3.0)
            
            if self.data.ndim == 3:
                result: NDArray = np.zeros_like(image)
                for i in range(3):
                    result[:, :, i] = ((1 - mask_blurred) * image[:, :, i] + 
                                      mask_blurred * background[:, :, i]).astype(np.uint8)
            else:
                result: NDArray = ((1 - mask_blurred) * image + mask_blurred * background).astype(np.uint8)
            
            self.progress.emit(100)
            self.finished.emit(image, mask, mask_eroded, result)
            
        except Exception as e:
            self.error.emit(f"Erreur de traitement: {str(e)}")

# Widget de comparaison avant/après avec barre glissante
class BeforeAfterWidget(QWidget):
    def __init__(self: BeforeAfterWidget, parent: QWidget = None):
        super().__init__(parent)
        self.before_image: QPixmap = None
        self.after_image: QPixmap = None
        self.slider_pos: float = 0.5  # Position de la barre (0.0 à 1.0)
        self.dragging: bool = False
        self.setMinimumSize(600, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        
    def set_images(self: BeforeAfterWidget, before: NDArray, after: NDArray):
        """Définit les images avant et après"""
        self.before_image = self._ndarray_to_pixmap(before)
        self.after_image = self._ndarray_to_pixmap(after)
        self.update()
        
    def _ndarray_to_pixmap(self: BeforeAfterWidget, image_data: NDArray) -> QPixmap:
        """Convertit un NDArray en QPixmap"""
        if image_data is None or image_data.size == 0:
            return None
            
        image_data = np.ascontiguousarray(image_data)
        
        if len(image_data.shape) == 2:
            height, width = image_data.shape
            q_image: QImage = QImage(image_data.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        else:
            height, width, channels = image_data.shape
            q_image: QImage = QImage(image_data.tobytes(), width, height, channels * width, QImage.Format.Format_RGB888)
            
        return QPixmap.fromImage(q_image)
        
    def paintEvent(self, event):
        """Dessine les images avec la barre de séparation"""
        painter: QPainter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.before_image is None or self.after_image is None:
            # Afficher un message si pas d'images
            painter.fillRect(self.rect(), QColor(250, 250, 250))
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "Chargez une image pour voir la comparaison")
            return
        
        # Calculer la taille d'affichage
        widget_width: int = self.width()
        widget_height: int = self.height()
        
        # Redimensionner les images pour s'adapter au widget
        scaled_before: QPixmap = self.before_image.scaled(
            widget_width, widget_height, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        scaled_after: QPixmap = self.after_image.scaled(
            widget_width, widget_height, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Centrer les images
        x_offset: int = (widget_width - scaled_before.width()) // 2
        y_offset: int = (widget_height - scaled_before.height()) // 2
        
        # Position de la barre en pixels
        slider_x: int = int(x_offset + scaled_before.width() * self.slider_pos)
        
        # Dessiner l'image "après" complète
        painter.drawPixmap(x_offset, y_offset, scaled_after)
        
        # Créer un clip pour l'image "avant" (partie gauche)
        painter.setClipRect(0, 0, slider_x, widget_height)
        painter.drawPixmap(x_offset, y_offset, scaled_before)
        
        # Désactiver le clipping pour dessiner la barre
        painter.setClipping(False)
        
        # Dessiner la barre de séparation
        pen: QPen = QPen(QColor(255, 255, 255), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(slider_x, 0, slider_x, widget_height)
        
        # Dessiner les poignées
        handle_size: int = 40
        handle_y: int = widget_height // 2
        
        # Ombre pour la poignée
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 50))
        painter.drawEllipse(slider_x - handle_size//2 + 2, handle_y - handle_size//2 + 2, 
                          handle_size, handle_size)
        
        # Poignée blanche
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(slider_x - handle_size//2, handle_y - handle_size//2, 
                          handle_size, handle_size)
        
        # Flèches sur la poignée
        painter.setPen(QPen(QColor(74, 144, 226), 3))
        arrow_size: int = 10
        # Flèche gauche
        painter.drawLine(slider_x - arrow_size, handle_y, slider_x - 3, handle_y - 7)
        painter.drawLine(slider_x - arrow_size, handle_y, slider_x - 3, handle_y + 7)
        # Flèche droite
        painter.drawLine(slider_x + arrow_size, handle_y, slider_x + 3, handle_y - 7)
        painter.drawLine(slider_x + arrow_size, handle_y, slider_x + 3, handle_y + 7)
        
        # Dessiner les labels "AVANT" et "APRÈS"
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        
        # Label AVANT (en haut à gauche)
        painter.drawText(x_offset + 20, y_offset + 30, "AVANT")
        
        # Label APRÈS (en haut à droite)
        text_width: int = painter.fontMetrics().horizontalAdvance("APRÈS")
        painter.drawText(x_offset + scaled_after.width() - text_width - 20, y_offset + 30, "APRÈS")
        
    def mousePressEvent(self, event: QMouseEvent):
        """Commence le glissement de la barre"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self._update_slider_pos(event.pos().x())
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Met à jour la position de la barre pendant le glissement"""
        if self.dragging:
            self._update_slider_pos(event.pos().x())
        # Changer le curseur près de la barre
        elif self.before_image is not None:
            slider_x: int = int(self.width() * self.slider_pos)
            if abs(event.pos().x() - slider_x) < 20:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Termine le glissement de la barre"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            
    def _update_slider_pos(self, x: int):
        """Met à jour la position du slider"""
        if self.before_image is None:
            return
            
        widget_width: int = self.width()
        scaled_width: int = min(self.before_image.width(), widget_width)
        x_offset: int = (widget_width - scaled_width) // 2
        
        # Limiter la position entre les bords de l'image
        self.slider_pos = max(0.0, min(1.0, (x - x_offset) / scaled_width))
        self.update()

# Interface utilisateur principale
class ImageCard(QFrame):
    def __init__(self: ImageCard, title: str, parent: QWidget = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setup_ui(title)
        
    def setup_ui(self: ImageCard, title: str):
        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Titre
        title_label: QLabel = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Zone d'image
        self.image_label: QLabel = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(280, 280)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.image_label)
        
        # Info
        self.info_label: QLabel = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.info_label)
        
    def set_image(self: ImageCard, image_data: NDArray):
        if image_data is None or image_data.size == 0:
            return
            
        image_data = np.ascontiguousarray(image_data)
        
        if len(image_data.shape) == 2:
            height, width = image_data.shape
            q_image: QImage = QImage(image_data.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        else:
            height, width, channels = image_data.shape
            q_image: QImage = QImage(image_data.tobytes(), width, height, channels * width, QImage.Format.Format_RGB888)
            
        pixmap: QPixmap = QPixmap.fromImage(q_image)
        scaled: QPixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                               Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        
    def set_info(self: ImageCard, text: str):
        self.info_label.setText(text)

# Interface principale
class StarReductionGUI(QMainWindow):
    def __init__(self: StarReductionGUI):
        super().__init__()
        self.data: NDArray = None
        self.images: dict[str, NDArray] = {}
        self.processing_thread: QThread = None

        self.processing_timer: QTimer = QTimer()
        self.processing_timer.setSingleShot(True)
        self.processing_timer.timeout.connect(self._process_image)
        
        self.setup_ui()
        self.apply_style()
        
    def setup_ui(self: StarReductionGUI):
        self.setWindowTitle("Réduction d'Étoiles")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central avec scroll
        central: QWidget = QWidget()
        self.setCentralWidget(central)
        central_layout: QVBoxLayout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll Area
        scroll_area: QScrollArea = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        central_layout.addWidget(scroll_area)
        
        # Contenu scrollable
        scroll_content: QWidget = QWidget()
        scroll_area.setWidget(scroll_content)
        main_layout: QVBoxLayout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        # ===== PREMIÈRE INTERFACE: Vue grille =====
        grid_section: QFrame = QFrame()
        grid_section.setFrameStyle(QFrame.Shape.StyledPanel)
        grid_layout: QVBoxLayout = QVBoxLayout(grid_section)
        grid_layout.setSpacing(20)
        grid_layout.setContentsMargins(25, 25, 25, 25)
        
        # En-tête
        header: QFrame = QFrame()
        header_layout: QVBoxLayout = QVBoxLayout(header)
        
        title: QLabel = QLabel("Réduction d'Étoiles")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        self.status_label: QLabel = QLabel("Chargez une image FITS pour commencer")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.status_label)
        
        grid_layout.addWidget(header)
        
        # Bouton de chargement
        load_btn: QPushButton = QPushButton("📁 Charger un fichier FITS")
        load_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        load_btn.setMinimumHeight(50)
        load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        load_btn.clicked.connect(self.load_file)
        grid_layout.addWidget(load_btn)
        
        # Contrôles
        controls: QFrame = QFrame()
        controls_layout: QVBoxLayout = QVBoxLayout(controls)
        controls_layout.setSpacing(15)
        
        # Intensité
        intensity_layout: QHBoxLayout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensité :"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(1, 19)
        self.intensity_slider.setValue(9)
        self.intensity_slider.setSingleStep(2)
        self.intensity_slider.valueChanged.connect(self.schedule_process)
        intensity_layout.addWidget(self.intensity_slider)
        self.intensity_label: QLabel = QLabel("9")
        self.intensity_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.intensity_label.setMinimumWidth(30)
        intensity_layout.addWidget(self.intensity_label)
        controls_layout.addLayout(intensity_layout)
        
        # Sensibilité
        sensitivity_layout: QHBoxLayout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Sensibilité :"))
        self.sensitivity_slider: QSlider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(5, 50)
        self.sensitivity_slider.setValue(10)
        self.sensitivity_slider.valueChanged.connect(self.schedule_process)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label: QLabel = QLabel("1.0")
        self.sensitivity_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.sensitivity_label.setMinimumWidth(30)
        sensitivity_layout.addWidget(self.sensitivity_label)
        controls_layout.addLayout(sensitivity_layout)
        
        controls.setMaximumHeight(120)
        grid_layout.addWidget(controls)
        
        # Cartes d'images
        cards_layout: QHBoxLayout = QHBoxLayout()
        cards_layout.setSpacing(15)
        
        self.card_original: ImageCard = ImageCard("Original")
        self.card_mask: ImageCard = ImageCard("Masque")
        self.card_eroded: ImageCard = ImageCard("Érodé")
        self.card_result: ImageCard = ImageCard("Résultat")
        
        cards_layout.addWidget(self.card_original)
        cards_layout.addWidget(self.card_mask)
        cards_layout.addWidget(self.card_eroded)
        cards_layout.addWidget(self.card_result)
        
        grid_layout.addLayout(cards_layout)
        
        # Ajouter la section grille au layout principal
        main_layout.addWidget(grid_section)
        
        # Séparateur
        separator: QLabel = QLabel("─" * 100)
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        separator.setFont(QFont("Segoe UI", 10))
        separator.setStyleSheet("color: #cccccc; padding: 20px;")
        main_layout.addWidget(separator)
        
        # ===== DEUXIÈME INTERFACE: Comparaison avant/après =====
        comparison_section: QFrame = QFrame()
        comparison_section.setFrameStyle(QFrame.Shape.StyledPanel)
        comparison_section_layout: QVBoxLayout = QVBoxLayout(comparison_section)
        comparison_section_layout.setSpacing(20)
        comparison_section_layout.setContentsMargins(25, 25, 25, 25)
        
        # Titre de la section
        comparison_title: QLabel = QLabel("Comparaison Avant / Après")
        comparison_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        comparison_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        comparison_section_layout.addWidget(comparison_title)
        
        comparison_subtitle: QLabel = QLabel("Glissez la barre pour comparer l'image originale et le résultat")
        comparison_subtitle.setFont(QFont("Segoe UI", 11))
        comparison_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        comparison_subtitle.setStyleSheet("color: #666666; padding-bottom: 10px;")
        comparison_section_layout.addWidget(comparison_subtitle)
        
        # Widget de comparaison
        self.comparison_widget: BeforeAfterWidget = BeforeAfterWidget()
        comparison_section_layout.addWidget(self.comparison_widget)
        
        # Info
        self.comparison_info_label: QLabel = QLabel("")
        self.comparison_info_label.setFont(QFont("Segoe UI", 10))
        self.comparison_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        comparison_section_layout.addWidget(self.comparison_info_label)
        
        # Ajouter la section comparaison au layout principal
        main_layout.addWidget(comparison_section)
        
        # Bouton de sauvegarde
        save_btn: QPushButton = QPushButton("Sauvegarder les résultats")
        save_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        save_btn.setMinimumHeight(50)
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self.save_results)
        main_layout.addWidget(save_btn)
        
        # Désactiver les contrôles
        self.set_controls_enabled(False)
        
    def apply_style(self: StarReductionGUI):
        self.setStyleSheet("""
            QMainWindow {
                background: #f5f5f5;
            }
            QScrollArea {
                border: none;
                background: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background: #4a90e2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #357abd;
            }
            QPushButton:pressed {
                background: #2868a8;
            }
            QPushButton:disabled {
                background: #cccccc;
            }
            QFrame {
                background: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            ImageCard QLabel {
                background: #fafafa;
                border-radius: 6px;
                padding: 10px;
                border: 1px solid #e8e8e8;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #4a90e2;
                border-radius: 3px;
            }
        """)
        
    def set_controls_enabled(self: StarReductionGUI, enabled: bool):
        self.intensity_slider.setEnabled(enabled)
        self.sensitivity_slider.setEnabled(enabled)
        
    def load_file(self: StarReductionGUI):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir FITS", "./examples", 
                                              "FITS (*.fits *.fit)")
        if path:
            try:
                with fits.open(path) as hdul:
                    self.data: NDArray = hdul[0].data
                
                filename: str = os.path.basename(path)
                self.status_label.setText(f"{filename}")
                self.set_controls_enabled(True)
                self._process_image()
                
            except Exception as e:
                self.status_label.setText(f"Erreur: {str(e)}")
                
    def schedule_process(self: StarReductionGUI):
        if self.data is None:
            return
        
        # Mise à jour des labels
        intensity: int = self.intensity_slider.value()
        if intensity % 2 == 0:
            intensity += 1
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(intensity)
            self.intensity_slider.blockSignals(False)
        self.intensity_label.setText(str(intensity))
        
        sensitivity: float = self.sensitivity_slider.value() / 10.0
        self.sensitivity_label.setText(f"{sensitivity:.1f}")
        
        self.processing_timer.start(500)
        
    def _process_image(self: StarReductionGUI):
        if self.data is None:
            return
            
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.quit()
            self.processing_thread.wait()
            
        kernel: int = self.intensity_slider.value()
        if kernel % 2 == 0:
            kernel += 1
        threshold: float = self.sensitivity_slider.value() / 10.0
        
        self.processing_thread = ImageProcessingThread(self.data, kernel, threshold)
        self.processing_thread.finished.connect(self.display_results)
        self.processing_thread.error.connect(self.handle_error)
        self.processing_thread.start()
        
    def display_results(self: StarReductionGUI, original: NDArray, mask: NDArray, eroded: NDArray, result: NDArray):
        self.images = {
            'original': original,
            'mask': mask,
            'eroded': eroded,
            'result': result
        }
        
        # Mettre à jour la vue grille
        self.card_original.set_image(original)
        self.card_mask.set_image(mask)
        self.card_eroded.set_image(eroded)
        self.card_result.set_image(result)
        
        stars: int = np.count_nonzero(mask)
        self.card_mask.set_info(f"{stars:,} pixels")
        
        # Mettre à jour la vue comparaison
        self.comparison_widget.set_images(original, result)
        self.comparison_info_label.setText(f"{stars:,} pixels d'étoiles détectés")
        
        
    def handle_error(self: StarReductionGUI, msg: str):
        self.status_label.setText(f"{msg}")
        
    def save_results(self: StarReductionGUI):
        if not self.images:
            return
            
        try:
            os.makedirs('./results', exist_ok=True)
            
            cv.imwrite('./results/original.png', self.images['original'])
            cv.imwrite('./results/mask.png', self.images['mask'])
            cv.imwrite('./results/eroded.png', self.images['eroded'])
            cv.imwrite('./results/result.png', self.images['result'])
            
            self.status_label.setText("Sauvegardé dans ./results/")
            
        except Exception as e:
            self.status_label.setText(f"Erreur: {str(e)}")


def main():
    app: QApplication = QApplication(sys.argv)
    app.setStyle('Fusion')
    window: StarReductionGUI = StarReductionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()