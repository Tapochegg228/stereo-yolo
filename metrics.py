#!/usr/bin/env python3
"""
metrics.py — Сбор и экспорт метрик производительности.

Метрики:
- FPS (average, min, max)
- Latency (per frame, ms)
- Depth accuracy (std dev, consistency)
- Detection rate
- Export to CSV
"""

import time
import csv
import os
import numpy as np
from collections import deque
from datetime import datetime


class MetricsCollector:
    """Сбор метрик производительности в реальном времени."""
    
    def __init__(self, window_size=100, export_dir="metrics_output"):
        """
        Args:
            window_size: Размер скользящего окна для расчёта средних
            export_dir: Директория для экспорта CSV
        """
        self.window_size = window_size
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        
        # FPS и Latency
        self.frame_times = deque(maxlen=window_size)
        self.total_frames = 0
        self.start_time = time.time()
        self._frame_start = 0.0
        
        # YOLO inference times
        self.yolo_times = deque(maxlen=window_size)
        
        # Depth per marker (marker_id -> deque of depths)
        self.depth_history = {}
        
        # Detection
        self.detection_counts = deque(maxlen=window_size)
        
        # Per-frame log for CSV
        self._frame_log = []
    
    def frame_start(self):
        """Вызвать в начале обработки кадра."""
        self._frame_start = time.perf_counter()
    
    def frame_end(self, detections_with_depth, yolo_time_ms=0.0):
        """
        Вызвать в конце обработки кадра.
        
        Args:
            detections_with_depth: list of dict с depth_mm
            yolo_time_ms: Время YOLO-инференса в мс
        """
        frame_time = (time.perf_counter() - self._frame_start) * 1000  # ms
        self.frame_times.append(frame_time)
        self.yolo_times.append(yolo_time_ms)
        self.total_frames += 1
        
        num_detections = len(detections_with_depth)
        self.detection_counts.append(num_detections)
        
        # Сохраняем глубину для каждого маркера
        for i, det in enumerate(detections_with_depth):
            marker_key = det.get("class_name", f"marker_{i}")
            depth = det.get("depth_mm", -1)
            
            if marker_key not in self.depth_history:
                self.depth_history[marker_key] = deque(maxlen=self.window_size)
            
            if depth > 0:
                self.depth_history[marker_key].append(depth)
        
        # Лог для CSV
        self._frame_log.append({
            "timestamp": time.time(),
            "frame": self.total_frames,
            "latency_ms": frame_time,
            "yolo_ms": yolo_time_ms,
            "num_detections": num_detections,
            "depths": {
                det.get("class_name", f"marker_{i}"): det.get("depth_mm", -1)
                for i, det in enumerate(detections_with_depth)
            }
        })
    
    # --- Getters ---
    
    def get_fps(self):
        """Текущий FPS (средний по окну)."""
        if not self.frame_times:
            return 0.0
        avg_ms = np.mean(self.frame_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0
    
    def get_fps_stats(self):
        """FPS: min, avg, max за окно."""
        if not self.frame_times:
            return {"min": 0, "avg": 0, "max": 0}
        
        times = np.array(self.frame_times)
        return {
            "min": float(1000.0 / np.max(times)) if np.max(times) > 0 else 0,
            "avg": float(1000.0 / np.mean(times)) if np.mean(times) > 0 else 0,
            "max": float(1000.0 / np.min(times)) if np.min(times) > 0 else 0,
        }
    
    def get_latency_ms(self):
        """Средняя задержка кадра (мс)."""
        if not self.frame_times:
            return 0.0
        return float(np.mean(self.frame_times))
    
    def get_yolo_latency_ms(self):
        """Средняя задержка YOLO-инференса (мс)."""
        if not self.yolo_times:
            return 0.0
        return float(np.mean(self.yolo_times))
    
    def get_depth_stability(self, marker_key=None):
        """
        Стабильность глубины (std dev в мм) за окно.
        
        Args:
            marker_key: Ключ конкретного маркера, или None для всех
        
        Returns:
            dict: {marker_key: {"mean_mm": ..., "std_mm": ..., "count": ...}}
        """
        result = {}
        
        targets = self.depth_history
        if marker_key and marker_key in self.depth_history:
            targets = {marker_key: self.depth_history[marker_key]}
        
        for key, depths in targets.items():
            if len(depths) > 1:
                arr = np.array(depths)
                result[key] = {
                    "mean_mm": float(np.mean(arr)),
                    "std_mm": float(np.std(arr)),
                    "count": len(arr)
                }
        
        return result
    
    def get_detection_rate(self):
        """Средний процент кадров с хотя бы одним обнаружением."""
        if not self.detection_counts:
            return 0.0
        return float(np.mean([1 if c > 0 else 0 for c in self.detection_counts])) * 100
    
    def get_summary(self):
        """Полная сводка метрик."""
        fps_stats = self.get_fps_stats()
        return {
            "total_frames": self.total_frames,
            "elapsed_seconds": time.time() - self.start_time,
            "fps": fps_stats,
            "latency_avg_ms": self.get_latency_ms(),
            "yolo_avg_ms": self.get_yolo_latency_ms(),
            "detection_rate_pct": self.get_detection_rate(),
            "depth_stability": self.get_depth_stability()
        }
    
    def get_overlay_text(self):
        """Текст для отображения поверх кадра."""
        fps = self.get_fps()
        latency = self.get_latency_ms()
        yolo_ms = self.get_yolo_latency_ms()
        det_rate = self.get_detection_rate()
        
        lines = [
            f"FPS: {fps:.1f}",
            f"Latency: {latency:.1f} ms",
            f"YOLO: {yolo_ms:.1f} ms",
            f"Det rate: {det_rate:.0f}%",
        ]
        
        # Стабильность глубины
        stability = self.get_depth_stability()
        for key, stats in stability.items():
            lines.append(
                f"{key}: {stats['mean_mm']:.0f}±{stats['std_mm']:.1f} mm"
            )
        
        return lines
    
    # --- Export ---
    
    def export_csv(self, filename=None):
        """Экспорт всех кадровых данных в CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.export_dir, f"metrics_{timestamp}.csv")
        
        if not self._frame_log:
            print("⚠️  Нет данных для экспорта")
            return
        
        # Собираем все уникальные ключи маркеров
        marker_keys = set()
        for entry in self._frame_log:
            marker_keys.update(entry["depths"].keys())
        marker_keys = sorted(marker_keys)
        
        headers = [
            "timestamp", "frame", "latency_ms", "yolo_ms", "num_detections"
        ] + [f"depth_{k}_mm" for k in marker_keys]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for entry in self._frame_log:
                row = {
                    "timestamp": f"{entry['timestamp']:.3f}",
                    "frame": entry["frame"],
                    "latency_ms": f"{entry['latency_ms']:.2f}",
                    "yolo_ms": f"{entry['yolo_ms']:.2f}",
                    "num_detections": entry["num_detections"],
                }
                for k in marker_keys:
                    depth = entry["depths"].get(k, -1)
                    row[f"depth_{k}_mm"] = f"{depth:.2f}" if depth > 0 else ""
                
                writer.writerow(row)
        
        print(f"\n💾 Метрики экспортированы: {filename}")
        print(f"   Кадров: {len(self._frame_log)}")
        print(f"   Маркеров: {len(marker_keys)}")
        
        return filename
    
    def print_summary(self):
        """Печать итоговой сводки в консоль."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("📊 ИТОГОВАЯ СВОДКА МЕТРИК")
        print("=" * 50)
        print(f"  Всего кадров: {summary['total_frames']}")
        print(f"  Время работы: {summary['elapsed_seconds']:.1f} сек")
        print(f"\n  FPS:")
        print(f"    Средний: {summary['fps']['avg']:.1f}")
        print(f"    Мин:     {summary['fps']['min']:.1f}")
        print(f"    Макс:    {summary['fps']['max']:.1f}")
        print(f"\n  Задержка:")
        print(f"    Кадр:     {summary['latency_avg_ms']:.1f} мс")
        print(f"    YOLO:     {summary['yolo_avg_ms']:.1f} мс")
        print(f"\n  Detection rate: {summary['detection_rate_pct']:.1f}%")
        
        if summary['depth_stability']:
            print(f"\n  Стабильность глубины:")
            for key, stats in summary['depth_stability'].items():
                print(f"    {key}: {stats['mean_mm']:.1f} ± {stats['std_mm']:.2f} мм "
                      f"({stats['count']} замеров)")
        
        print("=" * 50)
