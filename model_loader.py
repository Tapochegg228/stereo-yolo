#!/usr/bin/env python3
"""
model_loader.py — Загрузка и подготовка 3D-моделей (.obj) для AR-рендеринга.

Использует trimesh для парсинга OBJ файлов.
Нормализует модель: центрирует и масштабирует до единичного размера.
"""

import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError("Необходима библиотека trimesh: pip install trimesh")


class ModelData:
    """Данные загруженной 3D-модели."""
    
    def __init__(self, vertices, faces, face_normals, face_colors=None):
        """
        Args:
            vertices: np.array (N, 3) — вершины модели (нормализованные)
            faces: np.array (M, 3) — индексы вершин для каждой грани
            face_normals: np.array (M, 3) — нормали граней
            face_colors: np.array (M, 3) — цвета граней BGR (опционально)
        """
        self.vertices = vertices.astype(np.float64)
        self.faces = faces.astype(np.int32)
        self.face_normals = face_normals.astype(np.float64)
        
        if face_colors is not None:
            self.face_colors = face_colors.astype(np.uint8)
        else:
            # Цвет по умолчанию — серо-голубой
            self.face_colors = np.full((len(faces), 3), [180, 180, 200], dtype=np.uint8)
        
        self.num_vertices = len(vertices)
        self.num_faces = len(faces)
    
    def __repr__(self):
        return (f"ModelData(vertices={self.num_vertices}, "
                f"faces={self.num_faces})")


def load_model(path, normalize=True, max_faces=1500):
    """
    Загружает 3D-модель из файла (.obj, .stl, .ply и др.).
    
    Args:
        path: Путь к файлу модели
        normalize: Центрировать и масштабировать до единичного размера
        max_faces: Максимальное число граней (упрощает если больше)
    
    Returns:
        ModelData с вершинами, гранями, нормалями
    """
    print(f"🔧 Загрузка 3D модели: {path}")
    
    # Загрузка через trimesh
    mesh = trimesh.load(path, force='mesh')
    
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            if len(meshes) == 0:
                raise ValueError(f"Файл {path} не содержит 3D-геометрии")
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"Не удалось загрузить меш из {path}")
    
    print(f"   Вершины: {len(mesh.vertices)}")
    print(f"   Грани: {len(mesh.faces)}")
    
    # Упрощение меша если слишком много граней
    if max_faces and len(mesh.faces) > max_faces:
        print(f"   Упрощение: {len(mesh.faces)} → {max_faces} граней...")
        try:
            mesh = mesh.simplify_quadric_decimation(max_faces)
            print(f"   Результат: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
        except Exception as e:
            print(f"   ⚠️ Упрощение не удалось ({e}), используем оригинал")
    
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    if normalize:
        center = vertices.mean(axis=0)
        vertices -= center
        
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = extent.max()
        if max_extent > 1e-6:
            vertices /= max_extent
        
        print(f"   Нормализация: центрирована, scale={max_extent:.2f}")
    
    # Пересчитываем нормали граней
    mesh_normalized = trimesh.Trimesh(vertices=vertices, faces=faces)
    face_normals = np.array(mesh_normalized.face_normals, dtype=np.float64)
    
    # Извлекаем цвета граней (если есть)
    face_colors = None
    if mesh.visual and hasattr(mesh.visual, 'face_colors'):
        try:
            fc = np.array(mesh.visual.face_colors)
            if fc.shape[0] == len(faces) and fc.shape[1] >= 3:
                face_colors = fc[:, :3][:, ::-1].copy()
        except Exception:
            pass
    
    model = ModelData(vertices, faces, face_normals, face_colors)
    print(f"   ✅ Модель загружена: {model}")
    
    return model
