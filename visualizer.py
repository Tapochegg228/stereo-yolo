#!/usr/bin/env python3
"""
visualizer.py — Визуализация результатов детекции и глубины.

Отображает:
- Bounding box'ы маркеров
- Глубину (мм) для каждого маркера
- Метрики FPS, latency, detection rate
- Центроиды для подхода B
"""

import cv2
import numpy as np


# Цветовая палитра для маркеров (BGR)
MARKER_COLORS = [
    (0, 255, 0),     # Зелёный
    (255, 100, 0),   # Синий
    (0, 200, 255),   # Жёлтый
    (255, 0, 255),   # Розовый
    (0, 255, 255),   # Голубой
]


class Visualizer:
    """Визуализация детекции и глубины на кадре."""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
    
    def draw_detections(self, frame, results, metrics_text=None):
        """
        Рисует bbox'ы, глубину и метрики на кадре.
        
        Args:
            frame: BGR кадр для отрисовки (будет модифицирован)
            results: list of dict из StereoDepthEstimator.compute_depth_for_markers()
            metrics_text: list of str — строки метрик для overlay
        
        Returns:
            frame: Модифицированный кадр
        """
        for i, r in enumerate(results):
            color = MARKER_COLORS[i % len(MARKER_COLORS)]
            self._draw_marker(frame, r, color, i)
        
        # Метрики в левом верхнем углу
        if metrics_text:
            self._draw_metrics_overlay(frame, metrics_text)
        
        return frame
    
    def _draw_marker(self, frame, result, color, index):
        """Рисует один маркер с bbox и глубиной."""
        x1, y1, x2, y2 = [int(v) for v in result["bbox"]]
        
        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Глубина
        depth_mm = result.get("depth_mm", -1)
        conf = result.get("confidence", 0)
        method = result.get("method", "?")
        class_name = result.get("class_name", "marker")
        quality = result.get("quality", "")
        
        if depth_mm > 0:
            depth_m = depth_mm / 1000.0
            depth_text = f"{depth_m:.3f}m"
            depth_color = self._depth_color(depth_mm)
        else:
            depth_text = "N/A"
            depth_color = (0, 0, 255)
        
        # Основная метка сверху bbox
        label = f"{class_name} {conf:.0%}"
        label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        
        # Фон для текста
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                      (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                    self.font, self.font_scale, (0, 0, 0), self.thickness)
        
        # Глубина под bbox — крупным шрифтом
        depth_font_scale = 0.9
        depth_label = depth_text
        ds, _ = cv2.getTextSize(depth_label, self.font, depth_font_scale, 2)
        
        cx = (x1 + x2) // 2
        dy = y2 + ds[1] + 10
        
        # Фон
        cv2.rectangle(frame, (cx - ds[0]//2 - 5, y2 + 2),
                      (cx + ds[0]//2 + 5, dy + 5), (0, 0, 0), -1)
        cv2.putText(frame, depth_label, (cx - ds[0]//2, dy),
                    self.font, depth_font_scale, depth_color, 2)
        
        # Маленькая метка метода
        method_label = f"[{method}]"
        if quality:
            method_label += f" {quality}"
        cv2.putText(frame, method_label, (x1, y2 + ds[1] + 25),
                    self.font, 0.4, (180, 180, 180), 1)
        
        # Центроид (если есть данные от centroid метода)
        if "centroid_left" in result:
            cx_l, cy_l = result["centroid_left"]
            cv2.circle(frame, (int(cx_l), int(cy_l)), 4, (0, 0, 255), -1)
            cv2.circle(frame, (int(cx_l), int(cy_l)), 6, (255, 255, 255), 1)
    
    def _depth_color(self, depth_mm):
        """Цвет в зависимости от глубины (ближе = зелёный, дальше = красный)."""
        # Нормируем 200-2000 мм → 0-1
        t = min(1.0, max(0.0, (depth_mm - 200) / 1800.0))
        
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 50
        
        return (b, g, r)  # BGR
    
    def _draw_metrics_overlay(self, frame, lines):
        """Рисует полупрозрачную панель с метриками."""
        if not lines:
            return
        
        # Размер панели
        line_height = 25
        max_width = 0
        for line in lines:
            size, _ = cv2.getTextSize(line, self.font, 0.55, 1)
            max_width = max(max_width, size[0])
        
        panel_w = max_width + 20
        panel_h = line_height * len(lines) + 15
        
        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Текст
        for i, line in enumerate(lines):
            y = 25 + i * line_height
            
            # Цвет: FPS зелёный, остальное белое
            color = (0, 255, 0) if line.startswith("FPS") else (255, 255, 255)
            
            cv2.putText(frame, line, (15, y), self.font, 0.55, color, 1)
