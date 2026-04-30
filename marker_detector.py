#!/usr/bin/env python3
"""
marker_detector.py — Обёртка YOLOv8 для детектирования маркеров.

Загружает модель best.pt и запускает инференс через MPS бэкенд
на Apple Silicon для максимальной скорости.
"""

import numpy as np
import time

from ultralytics import YOLO


class MarkerDetector:
    """Детектор светоотражающих маркеров на базе YOLOv8."""
    
    def __init__(self, model_path="best.pt", confidence=0.5, device=None, imgsz=640):
        """
        Args:
            model_path: Путь к обученной модели YOLOv8
            confidence: Минимальная уверенность для детекции
            device: Устройство для инференса (None = автовыбор)
            imgsz: Размер входного изображения для YOLO
        """
        self.confidence = confidence
        self.imgsz = imgsz
        self.last_inference_ms = 0.0
        
        # Автовыбор устройства
        if device is None:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🤖 Загрузка YOLO модели: {model_path}")
        print(f"   Устройство: {self.device}")
        print(f"   Уверенность: {self.confidence}")
        print(f"   Размер входа: {self.imgsz}")
        
        self.model = YOLO(model_path)
        
        # Warm-up: первый инференс медленный (компиляция)
        print("   Прогрев модели...")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, conf=self.confidence, device=self.device, 
                          imgsz=self.imgsz, verbose=False)
        
        print(f"   ✅ Модель готова\n")
    
    def detect(self, frame):
        """
        Детектирует маркеры на одном кадре.
        
        Args:
            frame: BGR изображение (numpy array)
        
        Returns:
            list of dict: Каждый маркер содержит:
                - bbox: (x1, y1, x2, y2) в пикселях
                - center: (cx, cy) центр bbox
                - confidence: float
                - class_id: int
                - class_name: str
        """
        start = time.perf_counter()
        
        results = self.model.predict(
            frame, 
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )
        
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                confs = result.boxes.conf.cpu().numpy()   # (N,)
                classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    class_name = self.model.names.get(classes[i], f"class_{classes[i]}")
                    
                    detections.append({
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "center": (float(cx), float(cy)),
                        "confidence": float(confs[i]),
                        "class_id": int(classes[i]),
                        "class_name": class_name
                    })
        
        return detections
    
    def get_inference_time_ms(self):
        """Время последнего инференса в миллисекундах."""
        return self.last_inference_ms
