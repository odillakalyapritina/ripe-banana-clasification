import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QVBoxLayout, QPushButton, QFileDialog, 
                            QWidget, QComboBox, QSlider, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import os

class BananaDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Deteksi Pisang Matang")
        self.setGeometry(100, 100, 1000, 800)
        
        # Inisialisasi model
        self.model = None
        self.load_model()
        
        # Parameter deteksi
        self.confidence_threshold = 0.5
        self.class_names = ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe']  # Sesuaikan dengan kelas di model Anda
        
        # Setup UI
        self.init_ui()
        
        # Variabel state
        self.image = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detection_active = False
        self.current_mode = "Image"
    
    def load_model(self):
        try:
            # Ganti dengan path model Anda
            model_path = "best_banana.pt"  # Pastikan nama file model sesuai
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Error", 
                                    "Model file tidak ditemukan!\n"
                                    "Pastikan file model 'best_banana.pt' ada di direktori yang sama.")
                return
            
            self.model = YOLO(model_path)
            QMessageBox.information(self, "Info", "Model deteksi pisang berhasil dimuat!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal memuat model: {str(e)}")
    
    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Area tampilan
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: black;")
        
        # Kontrol mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Image", "Video", "Webcam"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        
        # Kontrol confidence
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.confidence_label = QLabel(f"Confidence Threshold: {self.confidence_threshold:.2f}")
        
        # Tombol kontrol
        self.btn_open = QPushButton("Buka File")
        self.btn_open.clicked.connect(self.open_file)
        
        self.btn_detect = QPushButton("Mulai Deteksi")
        self.btn_detect.clicked.connect(self.toggle_detection)
        self.btn_detect.setEnabled(False)
        
        self.btn_save = QPushButton("Simpan Hasil")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        
        # Layout
        layout.addWidget(self.mode_combo)
        layout.addWidget(self.image_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_slider)
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_detect)
        layout.addWidget(self.btn_save)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
    
    def change_mode(self, mode):
        self.current_mode = mode
        self.reset_state()
    
    def update_confidence(self, value):
        self.confidence_threshold = value / 100
        self.confidence_label.setText(f"Confidence Threshold: {self.confidence_threshold:.2f}")
    
    def open_file(self):
        self.reset_state()
        
        if self.current_mode == "Image":
            file_name, _ = QFileDialog.getOpenFileName(self, "Buka Gambar", "", 
                                                     "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_name:
                self.image = cv2.imread(file_name)
                self.display_image(self.image)
                self.btn_detect.setEnabled(True)
        
        elif self.current_mode == "Video":
            file_name, _ = QFileDialog.getOpenFileName(self, "Buka Video", "", 
                                                      "Video Files (*.mp4 *.avi *.mov)")
            if file_name:
                self.cap = cv2.VideoCapture(file_name)
                ret, frame = self.cap.read()
                if ret:
                    self.display_image(frame)
                    self.btn_detect.setEnabled(True)
    
    def reset_state(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.image = None
        self.detection_active = False
        self.btn_detect.setText("Mulai Deteksi")
        self.btn_save.setEnabled(False)
        self.image_label.clear()
        self.image_label.setStyleSheet("background-color: black;")
    
    def toggle_detection(self):
        if not self.model:
            QMessageBox.warning(self, "Peringatan", "Model belum dimuat!")
            return
            
        self.detection_active = not self.detection_active
        
        if self.detection_active:
            self.btn_detect.setText("Stop Deteksi")
            self.btn_save.setEnabled(True)
            
            if self.current_mode == "Image":
                self.detect_objects(self.image)
            else:
                if self.current_mode == "Webcam" and self.cap is None:
                    self.cap = cv2.VideoCapture(0)
                self.timer.start(30)
        else:
            self.btn_detect.setText("Mulai Deteksi")
            self.timer.stop()
    
    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                if self.detection_active:
                    frame = self.detect_objects(frame)
                self.display_image(frame)
            else:
                self.timer.stop()
                self.cap.release()
                self.cap = None
    
    def detect_objects(self, frame):
        if self.model is None:
            return frame
            
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Hitung jumlah setiap kelas
        counts = {class_name: 0 for class_name in self.class_names}
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id < len(self.class_names):
                    counts[self.class_names[class_id]] += 1
        
        # Tambahkan teks informasi
        info_text = " | ".join([f"{name}: {count}" for name, count in counts.items()])
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        self.display_image(annotated_frame)
        return annotated_frame
    
    def display_image(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def save_result(self):
        if self.current_mode == "Image" and self.image_label.pixmap():
            file_name, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar", "",
                                                     "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
            if file_name:
                self.image_label.pixmap().save(file_name)
        else:
            QMessageBox.information(self, "Info", "Fitur penyimpanan video belum tersedia")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BananaDetectionApp()
    window.show()
    sys.exit(app.exec_())