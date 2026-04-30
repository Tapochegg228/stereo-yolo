#!/usr/bin/env python3
"""
stereo_depth.py — Расчёт глубины маркеров по стереопаре.

Реализует три подхода:
  A) StereoSGBM на ROI — классический, надёжный
  B) Sub-pixel Centroid Matching — максимальная точность (основной)
  C) Гибридный — B с fallback на A

Улучшения v2:
  - Sub-pixel параболическая интерполяция template matching
  - Kalman-фильтр для временного сглаживания
  - Bias correction (линейная коррекция смещения)
"""

import cv2
import numpy as np
import os


class DepthKalmanFilter:
    """
    Kalman-фильтр для сглаживания глубины одного маркера.
    
    Состояние: [depth, velocity]
    Измерение: [depth]
    """
    
    def __init__(self, process_noise=0.5, measurement_noise=10.0):
        """
        Args:
            process_noise: Шум процесса (как быстро ожидаем изменение глубины)
            measurement_noise: Шум измерения (сколько шума в замерах, мм)
        """
        self.kf = cv2.KalmanFilter(2, 1)  # 2 состояния, 1 измерение
        
        # Матрица перехода: depth += velocity * dt
        self.kf.transitionMatrix = np.array([
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        
        # Матрица измерения: наблюдаем только depth
        self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        
        # Шум процесса
        self.kf.processNoiseCov = np.array([
            [process_noise, 0],
            [0, process_noise * 0.1]
        ], dtype=np.float32)
        
        # Шум измерения
        self.kf.measurementNoiseCov = np.array(
            [[measurement_noise]], dtype=np.float32
        )
        
        # Начальная ковариация (высокая неопределённость)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 1000
        
        self.initialized = False
        self.miss_count = 0   # Счётчик пропущенных кадров
        self.max_miss = 10    # Сброс после N пропусков
    
    def update(self, depth_mm):
        """
        Обновляет фильтр новым измерением.
        
        Args:
            depth_mm: Измеренная глубина в мм (или -1 если нет)
        
        Returns:
            float: Отфильтрованная глубина в мм
        """
        if depth_mm <= 0:
            self.miss_count += 1
            if self.miss_count > self.max_miss:
                self.initialized = False
            elif self.initialized:
                # Только предсказание (без коррекции)
                prediction = self.kf.predict()
                return float(prediction[0, 0])
            return -1.0
        
        self.miss_count = 0
        measurement = np.array([[np.float32(depth_mm)]])
        
        if not self.initialized:
            # Инициализация первым измерением
            self.kf.statePost = np.array(
                [[np.float32(depth_mm)], [np.float32(0)]]
            )
            self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 100
            self.initialized = True
            return depth_mm
        
        # Predict + Correct
        self.kf.predict()
        corrected = self.kf.correct(measurement)
        
        return float(corrected[0, 0])


class MarkerTracker:
    """
    Отслеживает маркеры между кадрами и применяет Kalman-фильтр.
    Идентифицирует маркеры по позиции (nearest-neighbor).
    """
    
    def __init__(self, max_distance_px=100, process_noise=0.5, measurement_noise=10.0):
        self.trackers = {}  # id -> {kalman, last_pos, last_depth}
        self.next_id = 0
        self.max_distance = max_distance_px
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def update(self, results):
        """
        Обновляет трекинг и применяет Kalman к результатам.
        
        Args:
            results: list of dict из compute_depth_for_markers()
        
        Returns:
            list of dict: Результаты с добавленным filtered_depth_mm
        """
        if not results:
            # Увеличиваем miss_count для всех трекеров
            for tid in list(self.trackers.keys()):
                self.trackers[tid]["kalman"].update(-1.0)
                if self.trackers[tid]["kalman"].miss_count > self.trackers[tid]["kalman"].max_miss:
                    del self.trackers[tid]
            return results
        
        # Позиции текущих детекций
        current_positions = []
        for r in results:
            cx = (r["bbox"][0] + r["bbox"][2]) / 2
            cy = (r["bbox"][1] + r["bbox"][3]) / 2
            current_positions.append((cx, cy))
        
        # Matching: назначаем каждую детекцию ближайшему трекеру
        assigned = set()
        assignments = {}  # result_idx -> tracker_id
        
        for i, pos in enumerate(current_positions):
            best_id = None
            best_dist = self.max_distance
            
            for tid, tracker in self.trackers.items():
                if tid in assigned:
                    continue
                dx = pos[0] - tracker["last_pos"][0]
                dy = pos[1] - tracker["last_pos"][1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            
            if best_id is not None:
                assignments[i] = best_id
                assigned.add(best_id)
        
        # Обновляем или создаём трекеры
        updated_results = []
        for i, r in enumerate(results):
            if i in assignments:
                tid = assignments[i]
            else:
                # Новый маркер
                tid = self.next_id
                self.next_id += 1
                self.trackers[tid] = {
                    "kalman": DepthKalmanFilter(
                        self.process_noise, self.measurement_noise
                    ),
                    "last_pos": current_positions[i],
                    "last_depth": -1.0
                }
            
            # Kalman update
            raw_depth = r.get("depth_mm", -1.0)
            filtered_depth = self.trackers[tid]["kalman"].update(raw_depth)
            
            self.trackers[tid]["last_pos"] = current_positions[i]
            self.trackers[tid]["last_depth"] = filtered_depth
            
            r_updated = dict(r)
            r_updated["raw_depth_mm"] = raw_depth
            r_updated["depth_mm"] = filtered_depth  # Заменяем на отфильтрованную
            r_updated["tracker_id"] = tid
            updated_results.append(r_updated)
        
        # Miss update для неназначенных трекеров
        for tid in list(self.trackers.keys()):
            if tid not in assigned:
                self.trackers[tid]["kalman"].update(-1.0)
                if self.trackers[tid]["kalman"].miss_count > self.trackers[tid]["kalman"].max_miss:
                    del self.trackers[tid]
        
        return updated_results


class StereoDepthEstimator:
    """
    Оценка глубины маркеров на основе стереопары.
    Загружает калибровочные данные и предоставляет три метода расчёта.
    """
    
    def __init__(self, calibration_file="calibration_data/stereo_calibration.npz"):
        """
        Args:
            calibration_file: Путь к .npz файлу с калибровочными данными
        """
        if not os.path.exists(calibration_file):
            raise FileNotFoundError(
                f"Файл калибровки не найден: {calibration_file}\n"
                f"Сначала запустите: python calibration.py"
            )
        
        print(f"📐 Загрузка калибровки: {calibration_file}")
        data = np.load(calibration_file)
        
        # Remap-карты для ректификации
        self.map_l1 = data["map_l1"]
        self.map_l2 = data["map_l2"]
        self.map_r1 = data["map_r1"]
        self.map_r2 = data["map_r2"]
        
        # Проекционная матрица левой камеры (для AR-рендеринга)
        self.P_l = data["P_l"]
        
        # Параметры для расчёта глубины
        self.focal_length_px = float(data["focal_length_px"])
        self.baseline_mm = float(data["baseline_mm"])
        
        # Дополнительные параметры
        self.image_size = tuple(data["image_size"])
        self.stereo_rms = float(data["stereo_rms_error"])
        
        print(f"   Focal length: {self.focal_length_px:.2f} px")
        print(f"   Baseline: {self.baseline_mm:.2f} мм")
        print(f"   Stereo RMS: {self.stereo_rms:.4f}")
        
        # Вычисляем диапазон диспаритетов для рабочих дистанций
        # d = f * B / Z
        self.min_distance_mm = 150.0   # Минимальная дистанция
        self.max_distance_mm = 2500.0  # Максимальная дистанция
        
        self.max_disparity = self.focal_length_px * self.baseline_mm / self.min_distance_mm
        self.min_disparity = self.focal_length_px * self.baseline_mm / self.max_distance_mm
        
        print(f"   Disparity range: {self.min_disparity:.1f} - {self.max_disparity:.1f} px "
              f"(for {self.min_distance_mm:.0f}-{self.max_distance_mm:.0f} mm)")
        
        # --- Bias correction ---
        # Линейная коррекция: Z_corrected = scale * Z_raw + offset
        # По умолчанию без коррекции. Устанавливается через set_bias_correction()
        self.bias_scale = 1.0
        self.bias_offset = 0.0
        self._load_bias_correction()
        
        print(f"   ✅ Калибровка загружена\n")
        
        # StereoSGBM для подхода A
        self._init_sgbm()
        
        # Kalman-фильтр трекер
        self.tracker = MarkerTracker(
            max_distance_px=100,
            process_noise=0.5,
            measurement_noise=10.0
        )
        
        # Temporal consistency для stereo_yolo matching
        # Хранит пары (left_centroid, right_centroid) предыдущего кадра
        self._prev_stereo_matches = []  # list of (lc, rc) tuples
    
    def get_projection_matrix(self):
        """Возвращает проекционную матрицу P_l для AR-рендеринга."""
        return self.P_l
    
    def _load_bias_correction(self):
        """Загружает bias correction из файла, если есть."""
        bias_file = "calibration_data/bias_correction.npz"
        if os.path.exists(bias_file):
            bias_data = np.load(bias_file)
            self.bias_scale = float(bias_data["scale"])
            self.bias_offset = float(bias_data["offset"])
            print(f"   Bias correction: scale={self.bias_scale:.4f}, offset={self.bias_offset:.2f} мм")
        else:
            print(f"   Bias correction: не задана (запустите calibrate_bias.py)")
    
    def set_bias_correction(self, measured_pairs):
        """
        Устанавливает bias correction по парам (реальное, измеренное).
        
        Args:
            measured_pairs: list of (Z_real_mm, Z_measured_mm)
        """
        if len(measured_pairs) < 2:
            if measured_pairs:
                real, meas = measured_pairs[0]
                self.bias_scale = real / meas
                self.bias_offset = 0.0
            return
        
        reals = np.array([p[0] for p in measured_pairs])
        measured = np.array([p[1] for p in measured_pairs])
        
        # Линейная регрессия: Z_real = scale * Z_measured + offset
        A = np.vstack([measured, np.ones(len(measured))]).T
        result = np.linalg.lstsq(A, reals, rcond=None)
        self.bias_scale, self.bias_offset = result[0]
        
        # Сохраняем
        np.savez("calibration_data/bias_correction.npz",
                 scale=self.bias_scale, offset=self.bias_offset,
                 pairs=np.array(measured_pairs))
        
        print(f"   Bias correction установлена: scale={self.bias_scale:.4f}, "
              f"offset={self.bias_offset:.2f} мм")
    
    def _apply_bias_correction(self, depth_mm):
        """Применяет bias correction к глубине."""
        if depth_mm <= 0:
            return depth_mm
        return self.bias_scale * depth_mm + self.bias_offset
    
    def _init_sgbm(self):
        """Инициализация StereoSGBM с безопасными параметрами."""
        # Ограничиваем numDisparities для SGBM, чтобы не было OOM
        sgbm_max_disp = min(self.max_disparity, 256)
        num_disparities = int(np.ceil(sgbm_max_disp / 16.0)) * 16
        num_disparities = max(num_disparities, 64)
        
        self.sgbm_num_disp = num_disparities
        self.sgbm_min_distance = self.focal_length_px * self.baseline_mm / num_disparities
        
        block_size = 5
        
        print(f"   SGBM numDisparities: {num_disparities} "
              f"(min dist: {self.sgbm_min_distance:.0f} mm)")
        
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def rectify(self, left, right):
        """
        Ректификация пары кадров.
        """
        left_rect = cv2.remap(left, self.map_l1, self.map_l2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map_r1, self.map_r2, cv2.INTER_LINEAR)
        return left_rect, right_rect
    
    def disparity_to_depth_mm(self, disparity):
        """
        Преобразует диспаритет (в пикселях) в глубину (в мм).
        Z = (focal_length * baseline) / disparity
        """
        if disparity <= 0.5:
            return -1.0
        depth = (self.focal_length_px * self.baseline_mm) / disparity
        # Фильтр разумности
        if depth < 50 or depth > 5000:
            return -1.0
        return depth
    
    # ============================
    # Sub-pixel утилиты
    # ============================
    
    def _subpixel_peak(self, correlation_map, peak_loc):
        """
        Параболическая sub-pixel интерполяция пика корреляции.
        
        Fit парабола через 3 точки вокруг пика по оси X (горизонталь = диспаритет).
        
        Args:
            correlation_map: 2D карта корреляции от matchTemplate
            peak_loc: (x, y) пик из minMaxLoc
        
        Returns:
            (x_subpixel, y_subpixel): уточнённые координаты пика
        """
        x, y = peak_loc
        h, w = correlation_map.shape[:2]
        
        x_sub = float(x)
        y_sub = float(y)
        
        # Sub-pixel по X (критично для диспаритета)
        if 1 <= x <= w - 2:
            left_val = float(correlation_map[y, x - 1])
            center_val = float(correlation_map[y, x])
            right_val = float(correlation_map[y, x + 1])
            
            denom = left_val - 2 * center_val + right_val
            if abs(denom) > 1e-6:
                dx = 0.5 * (left_val - right_val) / denom
                # Ограничиваем сдвиг до ±0.5 пикселя
                dx = max(-0.5, min(0.5, dx))
                x_sub = x + dx
        
        # Sub-pixel по Y (для проверки ректификации)
        if 1 <= y <= h - 2:
            top_val = float(correlation_map[y - 1, x])
            center_val = float(correlation_map[y, x])
            bottom_val = float(correlation_map[y + 1, x])
            
            denom = top_val - 2 * center_val + bottom_val
            if abs(denom) > 1e-6:
                dy = 0.5 * (top_val - bottom_val) / denom
                dy = max(-0.5, min(0.5, dy))
                y_sub = y + dy
        
        return (x_sub, y_sub)
    
    # ============================
    # Подход A: StereoSGBM на ROI
    # ============================
    
    def sgbm_depth(self, left_rect, right_rect, bbox, padding=20):
        """
        Расчёт глубины через StereoSGBM на области маркера.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = left_rect.shape[:2]
        
        num_disp = self.sgbm_num_disp
        
        roi_y1 = max(0, y1 - padding)
        roi_y2 = min(h, y2 + padding)
        
        required_width = num_disp + (x2 - x1) + padding * 2
        
        roi_x1 = max(0, x1 - padding - num_disp)
        roi_x2 = min(w, x2 + padding)
        
        actual_width = roi_x2 - roi_x1
        if actual_width < num_disp + 20:
            expand = num_disp + 20 - actual_width
            roi_x2 = min(w, roi_x2 + expand)
            actual_width = roi_x2 - roi_x1
        
        if actual_width < num_disp + 10 or (roi_y2 - roi_y1) < 5:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "sgbm"}
        
        left_roi = left_rect[roi_y1:roi_y2, roi_x1:roi_x2]
        right_roi = right_rect[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if len(left_roi.shape) == 3:
            left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_roi
            right_gray = right_roi
        
        try:
            disparity_map = self.sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
        except cv2.error:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "sgbm"}
        
        marker_x1 = max(0, x1 - roi_x1)
        marker_y1 = max(0, y1 - roi_y1)
        marker_x2 = min(disparity_map.shape[1], x2 - roi_x1)
        marker_y2 = min(disparity_map.shape[0], y2 - roi_y1)
        
        if marker_x2 <= marker_x1 or marker_y2 <= marker_y1:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "sgbm"}
        
        marker_disp = disparity_map[marker_y1:marker_y2, marker_x1:marker_x2]
        valid = marker_disp[marker_disp > 0]
        
        if len(valid) == 0:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "sgbm"}
        
        median_disp = float(np.median(valid))
        depth_mm = self.disparity_to_depth_mm(median_disp)
        depth_mm = self._apply_bias_correction(depth_mm)
        
        return {
            "depth_mm": depth_mm,
            "disparity": median_disp,
            "method": "sgbm"
        }
    
    # =====================================
    # Подход B: Sub-pixel Centroid Matching
    # =====================================
    
    def centroid_depth(self, left_rect, right_rect, bbox, padding=15, debug=False):
        """
        Расчёт глубины через поиск яркого маркера в обоих кадрах.
        
        Алгоритм (v2 — с sub-pixel):
        1. Вырезаем ROI маркера из ЛЕВОГО кадра (bbox от YOLO)
        2. Находим centroid яркой области в левом ROI (sub-pixel)
        3. В ПРАВОМ кадре ищем маркер через template matching
        4. Sub-pixel параболическая интерполяция пика корреляции
        5. Centroid в правом кадре (blob detection)
        6. Диспаритет = x_left - x_right (true sub-pixel)
        7. Bias correction + Z = f * B / d
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = left_rect.shape[:2]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        if bbox_w < 3 or bbox_h < 3:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        # === ШАГ 1: Centroid в левом кадре (надёжно, т.к. bbox от YOLO) ===
        l_y1 = max(0, y1 - padding)
        l_y2 = min(h, y2 + padding)
        l_x1 = max(0, x1 - padding)
        l_x2 = min(w, x2 + padding)
        
        left_roi = left_rect[l_y1:l_y2, l_x1:l_x2]
        if len(left_roi.shape) == 3:
            left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            left_gray = left_roi.astype(np.float64)
        
        # Пороговая фильтрация — оставляем яркие пиксели
        left_max = left_gray.max()
        if left_max < 30:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        left_thresh = left_gray.copy()
        left_thresh[left_thresh < left_max * 0.5] = 0
        
        cx_left_local, cy_left_local = self._intensity_weighted_centroid(left_thresh)
        if cx_left_local < 0:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        # Глобальные координаты центроида в левом кадре
        global_cx_left = l_x1 + cx_left_local
        global_cy_left = l_y1 + cy_left_local
        
        # === ШАГ 2: Поиск маркера в правом кадре вдоль эпиполярной линии ===
        search_half_h = bbox_h // 2 + padding
        r_y1 = max(0, int(global_cy_left) - search_half_h)
        r_y2 = min(h, int(global_cy_left) + search_half_h)
        
        max_disp_search = int(self.max_disparity) + 50
        min_disp_search = max(0, int(self.min_disparity) - 10)
        
        r_x1 = max(0, int(global_cx_left) - max_disp_search)
        r_x2 = max(0, int(global_cx_left) - min_disp_search + bbox_w)
        
        if r_x2 <= r_x1 or r_x2 - r_x1 < 5 or r_y2 - r_y1 < 5:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        right_search = right_rect[r_y1:r_y2, r_x1:r_x2]
        if len(right_search.shape) == 3:
            right_gray = cv2.cvtColor(right_search, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            right_gray = right_search.astype(np.float64)
        
        # Создаём шаблон из левого bbox
        template_y1 = max(0, y1 - l_y1)
        template_y2 = min(left_gray.shape[0], y2 - l_y1)
        template_x1 = max(0, x1 - l_x1)
        template_x2 = min(left_gray.shape[1], x2 - l_x1)
        
        template = left_gray[template_y1:template_y2, template_x1:template_x2]
        
        if template.shape[0] < 3 or template.shape[1] < 3:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        if (right_gray.shape[0] < template.shape[0] or 
            right_gray.shape[1] < template.shape[1]):
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        # === ШАГ 3: Template matching + sub-pixel пик ===
        # TM_CCOEFF (без нормализации): в тёмных областях значения ~0,
        # маркер даёт высокий пик. Нет деления на локальную дисперсию → нет шумовых артефактов.
        result_tm = cv2.matchTemplate(
            right_gray.astype(np.float32), 
            template.astype(np.float32), 
            cv2.TM_CCOEFF
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result_tm)
        
        if max_val <= 0:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        # Sub-pixel уточнение пика корреляции (параболическая интерполяция)
        sub_x, sub_y = self._subpixel_peak(result_tm, max_loc)
        
        # Глобальные координаты центра matching-окна в правом кадре (sub-pixel!)
        global_match_cx_right = r_x1 + sub_x + template.shape[1] / 2.0
        global_match_cy_right = r_y1 + sub_y + template.shape[0] / 2.0
        
        # === ШАГ 4: Уточнение через centroid в правом ROI ===
        match_x_int = int(round(sub_x))
        rc_x1 = max(0, match_x_int - padding)
        rc_y1 = max(0, int(round(sub_y)) - padding)
        rc_x2 = min(right_gray.shape[1], match_x_int + template.shape[1] + padding)
        rc_y2 = min(right_gray.shape[0], int(round(sub_y)) + template.shape[0] + padding)
        
        right_marker_roi = right_gray[rc_y1:rc_y2, rc_x1:rc_x2]
        
        right_max = right_marker_roi.max()
        if right_max < 30:
            global_cx_right = global_match_cx_right
            global_cy_right = global_match_cy_right
        else:
            right_thresh = right_marker_roi.copy()
            right_thresh[right_thresh < right_max * 0.5] = 0
            
            cx_right_local, cy_right_local = self._intensity_weighted_centroid(right_thresh)
            if cx_right_local < 0:
                global_cx_right = global_match_cx_right
                global_cy_right = global_match_cy_right
            else:
                global_cx_right = r_x1 + rc_x1 + cx_right_local
                global_cy_right = r_y1 + rc_y1 + cy_right_local
        
        # Проверка y-координат (ректификация)
        y_diff = abs(global_cy_left - global_cy_right)
        
        # Sub-pixel диспаритет
        disparity = global_cx_left - global_cx_right
        
        if disparity < self.min_disparity * 0.5 or disparity > self.max_disparity * 1.5:
            return {"depth_mm": -1.0, "disparity": 0.0, "method": "centroid"}
        
        depth_mm = self.disparity_to_depth_mm(disparity)
        depth_mm = self._apply_bias_correction(depth_mm)
        
        return {
            "depth_mm": depth_mm,
            "disparity": float(disparity),
            "centroid_left": (float(global_cx_left), float(global_cy_left)),
            "centroid_right": (float(global_cx_right), float(global_cy_right)),
            "y_diff": float(y_diff),
            "match_quality": float(max_val),
            "method": "centroid",
            # Debug данные (только при debug=True)
            **({"_dbg_template": template,
                "_dbg_corr_map": result_tm,
                "_dbg_search_region": (r_x1, r_y1, r_x2, r_y2),
                "_dbg_match_loc": (float(sub_x + r_x1), float(sub_y + r_y1)),
                "_dbg_left_roi_origin": (l_x1, l_y1),
            } if debug else {})
        }
    
    def _intensity_weighted_centroid(self, roi_gray):
        """
        Вычисляет intensity-weighted centroid для ROI.
        
        x_c = Σ(x * I(x,y)) / Σ(I(x,y))
        y_c = Σ(y * I(x,y)) / Σ(I(x,y))
        
        Returns:
            tuple: (cx, cy) sub-pixel координаты, или (-1, -1)
        """
        total_intensity = roi_gray.sum()
        
        if total_intensity < 1e-6:
            return (-1.0, -1.0)
        
        h, w = roi_gray.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        cx = float(np.sum(x_coords * roi_gray) / total_intensity)
        cy = float(np.sum(y_coords * roi_gray) / total_intensity)
        
        return (cx, cy)
    
    # ==========================
    # Подход C: Гибридный
    # ==========================
    
    def hybrid_depth(self, left_rect, right_rect, bbox, padding=15):
        """
        Гибридный подход: centroid как основной, SGBM как fallback.
        """
        result_centroid = self.centroid_depth(left_rect, right_rect, bbox, padding)
        
        if result_centroid["depth_mm"] > 0:
            depth = result_centroid["depth_mm"]
            y_diff = result_centroid.get("y_diff", 0)
            match_q = result_centroid.get("match_quality", 0)
            
            if y_diff < 3.0 and match_q > 0.5 and 100 < depth < 3000:
                result_centroid["quality"] = "good"
                return result_centroid
            elif y_diff < 5.0 and match_q > 0.3 and 100 < depth < 3000:
                result_centroid["quality"] = "acceptable"
                return result_centroid
        
        result_sgbm = self.sgbm_depth(left_rect, right_rect, bbox, padding)
        result_sgbm["quality"] = "fallback"
        
        return result_sgbm
    
    # =========================================
    # Подход D: Stereo YOLO (YOLO на обоих кадрах)
    # =========================================
    
    def _compute_centroid_in_bbox(self, frame, bbox, padding=15):
        """
        Вычисляет intensity-weighted centroid маркера внутри bbox.
        Возвращает (global_cx, global_cy) или None если не удалось.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        
        roi_y1 = max(0, y1 - padding)
        roi_y2 = min(h, y2 + padding)
        roi_x1 = max(0, x1 - padding)
        roi_x2 = min(w, x2 + padding)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = roi.astype(np.float64)
        
        max_val = gray.max()
        if max_val < 30:
            return None
        
        thresh = gray.copy()
        thresh[thresh < max_val * 0.5] = 0
        
        cx_local, cy_local = self._intensity_weighted_centroid(thresh)
        if cx_local < 0:
            return None
        
        return (roi_x1 + cx_local, roi_y1 + cy_local)
    
    def stereo_yolo_depth(self, left_rect, right_rect, left_detections, right_detections, padding=15):
        """
        Расчёт глубины используя YOLO-детекции из ОБОИХ кадров.
        
        Преимущество: нет путаницы между одинаковыми маркерами,
        т.к. каждый маркер однозначно детектирован в обоих кадрах.
        
        Алгоритм:
        1. Centroid каждого маркера в левом кадре (из YOLO bbox)
        2. Centroid каждого маркера в правом кадре (из YOLO bbox)  
        3. Сопоставление: ближайший по Y + валидный диспаритет
        4. Disparity = left_cx - right_cx → depth
        """
        results = []
        
        # ШАГ 1: Centroid для каждого левого маркера
        left_centroids = []
        for det in left_detections:
            c = self._compute_centroid_in_bbox(left_rect, det["bbox"], padding)
            left_centroids.append(c)
        
        # ШАГ 2: Centroid для каждого правого маркера  
        right_centroids = []
        for det in right_detections:
            c = self._compute_centroid_in_bbox(right_rect, det["bbox"], padding)
            right_centroids.append(c)
        
        # ШАГ 3: Оптимальное сопоставление с temporal consistency
        # Перебираем все возможные сопоставления, выбираем с min стоимостью.
        # Стоимость = сумма Y-расстояний + штраф за смену пар из предыдущего кадра.
        # Для N=3 это 6 вариантов, для N=5 — 120. Тривиально быстро.
        
        from itertools import permutations
        
        left_valid = [(i, lc) for i, lc in enumerate(left_centroids) if lc is not None]
        right_valid = [(j, rc) for j, rc in enumerate(right_centroids) if rc is not None]
        
        match_map = {}  # left_det_index → (right_centroid, disparity, y_diff)
        
        n = min(len(left_valid), len(right_valid))
        
        if n > 0:
            best_perm = None
            best_cost = float('inf')
            
            # Штраф за смену: для каждого левого маркера проверяем,
            # совпадает ли правый матч с предыдущим кадром
            SWITCH_PENALTY = 10.0  # px — гистерезис
            
            for perm in permutations(range(len(right_valid)), n):
                cost = 0.0
                valid = True
                
                for k in range(n):
                    li, lc = left_valid[k]
                    rj, rc = right_valid[perm[k]]
                    
                    disp = lc[0] - rc[0]
                    
                    # Диспаритет должен быть в допустимом диапазоне
                    if disp < self.min_disparity * 0.5 or disp > self.max_disparity * 1.5:
                        valid = False
                        break
                    
                    # Стоимость = расстояние по Y (эпиполярное ограничение)
                    cost += abs(lc[1] - rc[1])
                    
                    # Штраф за смену правого матча: ищем этот левый маркер
                    # в предыдущем кадре (по близости координат)
                    if self._prev_stereo_matches:
                        matched_prev = False
                        for prev_lc, prev_rc in self._prev_stereo_matches:
                            if abs(lc[0] - prev_lc[0]) < 50 and abs(lc[1] - prev_lc[1]) < 50:
                                # Нашли этот маркер в предыдущем кадре
                                # Если правый матч изменился — штраф
                                if abs(rc[0] - prev_rc[0]) > 20 or abs(rc[1] - prev_rc[1]) > 20:
                                    cost += SWITCH_PENALTY
                                matched_prev = True
                                break
                
                if valid and cost < best_cost:
                    best_cost = cost
                    best_perm = perm
            
            # Записываем лучшее сопоставление и сохраняем для следующего кадра
            current_matches = []
            if best_perm is not None:
                for k in range(n):
                    li, lc = left_valid[k]
                    rj, rc = right_valid[best_perm[k]]
                    disp = lc[0] - rc[0]
                    match_map[li] = (rc, disp, abs(lc[1] - rc[1]))
                    current_matches.append((lc, rc))
            
            self._prev_stereo_matches = current_matches
        
        # ШАГ 4: Формируем результаты
        for i, det in enumerate(left_detections):
            lc = left_centroids[i]
            
            if lc is None or i not in match_map:
                results.append({
                    **det,
                    "depth_mm": -1.0,
                    "disparity": 0.0,
                    "method": "stereo_yolo",
                })
                continue
            
            rc, disparity, y_diff = match_map[i]
            
            depth_mm = self.disparity_to_depth_mm(disparity)
            depth_mm = self._apply_bias_correction(depth_mm)
            
            results.append({
                **det,
                "depth_mm": depth_mm,
                "disparity": float(disparity),
                "centroid_left": (float(lc[0]), float(lc[1])),
                "centroid_right": (float(rc[0]), float(rc[1])),
                "y_diff": float(y_diff),
                "match_quality": 1.0,
                "method": "stereo_yolo",
            })
        
        return results
    
    def compute_depth_for_markers(self, left_rect, right_rect, detections, 
                                   method="hybrid", right_detections=None):
        """
        Вычисляет глубину для списка обнаруженных маркеров.
        Применяет Kalman-фильтр для сглаживания.
        
        Args:
            right_detections: Детекции из правого кадра (для метода stereo_yolo).
                             Если None, используются методы TM/SGBM.
        """
        # Метод stereo_yolo: используем YOLO на обоих кадрах
        if method == "stereo_yolo" and right_detections is not None:
            results = self.stereo_yolo_depth(
                left_rect, right_rect, detections, right_detections
            )
        else:
            depth_func = {
                "centroid": self.centroid_depth,
                "sgbm": self.sgbm_depth,
                "hybrid": self.hybrid_depth
            }.get(method, self.hybrid_depth)
            
            results = []
            
            for det in detections:
                try:
                    depth_info = depth_func(left_rect, right_rect, det["bbox"])
                except Exception as e:
                    depth_info = {"depth_mm": -1.0, "disparity": 0.0, 
                                "method": method, "error": str(e)}
                
                result = {**det, **depth_info}
                results.append(result)
        
        # Применяем Kalman-фильтр
        results = self.tracker.update(results)
        
        return results
