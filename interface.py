import sys
import os
import warnings
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QFileDialog, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
import cv2 as cv

warnings.filterwarnings('ignore', category=NoDetectionsWarning)


class ImageProcessingThread(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, data, kernel_size, threshold_multiplier):
        super().__init__()
        self.data = data
        self.kernel_size = kernel_size
        self.threshold_multiplier = threshold_multiplier
        
    def run(self):
        try:
            self.progress.emit(10)
            
            # Conversion en uint8
            if self.data.ndim == 3:
                data_transposed = np.transpose(self.data, (1, 2, 0)) if self.data.shape[0] == 3 else self.data
                image = np.zeros_like(data_transposed, dtype='uint8')
                for i in range(data_transposed.shape[2]):
                    channel = data_transposed[:, :, i]
                    image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
                data_gray = np.mean(data_transposed, axis=2)
            else:
                image = ((self.data - self.data.min()) / (self.data.max() - self.data.min()) * 255).astype('uint8')
                data_gray = self.data
                
            self.progress.emit(30)
            
            # Détection des étoiles
            mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=self.threshold_multiplier * std)
            sources = daofind(data_gray - median)
            
            self.progress.emit(50)
            
            if sources is None or len(sources) == 0:
                self.error.emit("Aucune étoile détectée")
                self.finished.emit(image, np.zeros_like(data_gray, dtype=np.uint8), 
                                 np.zeros_like(data_gray, dtype=np.uint8), image)
                return
            
            # Création du masque
            mask = np.zeros(data_gray.shape, dtype=np.uint8)
            for source in sources:
                x, y = int(source['xcentroid']), int(source['ycentroid'])
                radius = int(max(4, min(12, source['sharpness'] * 8 + source['peak'] / 2000)))
                cv.circle(mask, (x, y), radius, 255, -1)
            
            self.progress.emit(70)
            
            # Érosion du masque
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            mask_eroded = cv.erode(mask, kernel, iterations=1)
            
            # Inpainting
            background = cv.inpaint(image, mask_eroded, 3, cv.INPAINT_TELEA)
            
            self.progress.emit(85)
            
            # Mélange progressif
            mask_float = mask_eroded.astype(np.float32) / 255.0
            mask_blurred = cv.GaussianBlur(mask_float, (11, 11), 3.0)
            
            if self.data.ndim == 3:
                result = np.zeros_like(image)
                for i in range(3):
                    result[:, :, i] = ((1 - mask_blurred) * image[:, :, i] + 
                                      mask_blurred * background[:, :, i]).astype(np.uint8)
            else:
                result = ((1 - mask_blurred) * image + mask_blurred * background).astype(np.uint8)
            
            self.progress.emit(100)
            self.finished.emit(image, mask, mask_eroded, result)
            
        except Exception as e:
            self.error.emit(f"Erreur de traitement: {str(e)}")


class ImageCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setup_ui(title)
        
    def setup_ui(self, title):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Titre
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Zone d'image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(280, 280)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.image_label)
        
        # Info
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.info_label)
        
    def set_image(self, image_data):
        if image_data is None or image_data.size == 0:
            return
            
        image_data = np.ascontiguousarray(image_data)
        
        if len(image_data.shape) == 2:
            height, width = image_data.shape
            q_image = QImage(image_data.tobytes(), width, height, width, QImage.Format.Format_Grayscale8)
        else:
            height, width, channels = image_data.shape
            q_image = QImage(image_data.tobytes(), width, height, channels * width, QImage.Format.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                               Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        
    def set_info(self, text):
        self.info_label.setText(text)


class StarReductionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.images = {}
        self.processing_thread = None
        self.processing_timer = QTimer()
        self.processing_timer.setSingleShot(True)
        self.processing_timer.timeout.connect(self._process_image)
        
        self.setup_ui()
        self.apply_style()
        
    def setup_ui(self):
        self.setWindowTitle("Réduction d'Étoiles")
        self.setGeometry(100, 100, 1400, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        # En-tête
        header = QFrame()
        header_layout = QVBoxLayout(header)
        
        title = QLabel("Réduction d'Étoiles")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        self.status_label = QLabel("Chargez une image FITS pour commencer")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.status_label)
        
        main_layout.addWidget(header)
        
        # Bouton de chargement
        load_btn = QPushButton("Charger un fichier FITS")
        load_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        load_btn.setMinimumHeight(50)
        load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        load_btn.clicked.connect(self.load_file)
        main_layout.addWidget(load_btn)
        
        # Contrôles
        controls = QFrame()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(15)
        
        # Intensité
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensité:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(1, 19)
        self.intensity_slider.setValue(9)
        self.intensity_slider.setSingleStep(2)
        self.intensity_slider.valueChanged.connect(self.schedule_process)
        intensity_layout.addWidget(self.intensity_slider)
        self.intensity_label = QLabel("9")
        self.intensity_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.intensity_label.setMinimumWidth(30)
        intensity_layout.addWidget(self.intensity_label)
        controls_layout.addLayout(intensity_layout)
        
        # Sensibilité
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("Sensibilité:"))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(5, 50)
        self.sensitivity_slider.setValue(10)
        self.sensitivity_slider.valueChanged.connect(self.schedule_process)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label = QLabel("1.0")
        self.sensitivity_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.sensitivity_label.setMinimumWidth(30)
        sensitivity_layout.addWidget(self.sensitivity_label)
        controls_layout.addLayout(sensitivity_layout)
        
        controls.setMaximumHeight(120)
        main_layout.addWidget(controls)
        
        # Cartes d'images
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)
        
        self.card_original = ImageCard("Original")
        self.card_mask = ImageCard("Masque")
        self.card_eroded = ImageCard("Érodé")
        self.card_result = ImageCard("Résultat")
        
        cards_layout.addWidget(self.card_original)
        cards_layout.addWidget(self.card_mask)
        cards_layout.addWidget(self.card_eroded)
        cards_layout.addWidget(self.card_result)
        
        main_layout.addLayout(cards_layout, 1)
        
        # Bouton de sauvegarde
        save_btn = QPushButton("Sauvegarder les résultats")
        save_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        save_btn.setMinimumHeight(50)
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self.save_results)
        main_layout.addWidget(save_btn)
        
        # Désactiver les contrôles
        self.set_controls_enabled(False)
        
    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
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
        
    def set_controls_enabled(self, enabled):
        self.intensity_slider.setEnabled(enabled)
        self.sensitivity_slider.setEnabled(enabled)
        
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir FITS", "./examples", 
                                              "FITS (*.fits *.fit)")
        if path:
            try:
                with fits.open(path) as hdul:
                    self.data = hdul[0].data
                
                filename = os.path.basename(path)
                self.status_label.setText(f"{filename}")
                self.set_controls_enabled(True)
                self._process_image()
                
            except Exception as e:
                self.status_label.setText(f"Erreur: {str(e)}")
                
    def schedule_process(self):
        if self.data is None:
            return
        
        # Mise à jour des labels
        intensity = self.intensity_slider.value()
        if intensity % 2 == 0:
            intensity += 1
            self.intensity_slider.blockSignals(True)
            self.intensity_slider.setValue(intensity)
            self.intensity_slider.blockSignals(False)
        self.intensity_label.setText(str(intensity))
        
        sensitivity = self.sensitivity_slider.value() / 10.0
        self.sensitivity_label.setText(f"{sensitivity:.1f}")
        
        self.processing_timer.start(500)
        
    def _process_image(self):
        if self.data is None:
            return
            
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.quit()
            self.processing_thread.wait()
            
        kernel = self.intensity_slider.value()
        if kernel % 2 == 0:
            kernel += 1
        threshold = self.sensitivity_slider.value() / 10.0
        
        self.processing_thread = ImageProcessingThread(self.data, kernel, threshold)
        self.processing_thread.finished.connect(self.display_results)
        self.processing_thread.error.connect(self.handle_error)
        self.processing_thread.start()
        
    def display_results(self, original, mask, eroded, result):
        self.images = {
            'original': original,
            'mask': mask,
            'eroded': eroded,
            'result': result
        }
        
        self.card_original.set_image(original)
        self.card_mask.set_image(mask)
        self.card_eroded.set_image(eroded)
        self.card_result.set_image(result)
        
        stars = np.count_nonzero(mask)
        self.card_mask.set_info(f"{stars:,} pixels")
        
        
    def handle_error(self, msg):
        self.status_label.setText(f"{msg}")
        
    def save_results(self):
        if not self.images:
            return
            
        try:
            os.makedirs('./results', exist_ok=True)
            
            cv.imwrite('./results/original.png', self.images['original'])
            cv.imwrite('./results/mask.png', self.images['mask'])
            cv.imwrite('./results/eroded.png', self.images['eroded'])
            cv.imwrite('./results/result.png', self.images['result'], 
                      [cv.IMWRITE_PNG_COMPRESSION, 9])
            
            self.status_label.setText("Sauvegardé dans ./results/")
            
        except Exception as e:
            self.status_label.setText(f"Erreur: {str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = StarReductionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()