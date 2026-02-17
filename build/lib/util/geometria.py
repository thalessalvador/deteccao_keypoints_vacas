import math
import numpy as np
from typing import Tuple, Optional, Any
from .contratos import Bbox

def calcular_distancia(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    calcular_distancia: Calcula a distância euclidiana entre dois pontos (x, y).

    Args:
        p1 (Tuple[float, float]): Coordenadas (x, y) do ponto 1.
        p2 (Tuple[float, float]): Coordenadas (x, y) do ponto 2.

    Returns:
        float: Distância euclidiana entre p1 e p2.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calcular_angulo(p1: Tuple[float, float], p_vertex: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    calcular_angulo: Calcula o ângulo (em graus) formado por três pontos (A, V, B) onde V é o vértice.

    O ângulo retornado está no intervalo [0, 180].

    Args:
        p1 (Tuple[float, float]): Ponto A.
        p_vertex (Tuple[float, float]): Ponto Vértice.
        p3 (Tuple[float, float]): Ponto B.

    Returns:
        float: Ângulo em graus. Retorna NaN se algum ponto for coincidente com o vértice (divisão por zero na normalização).
    """
    # Vetores
    v1 = np.array([p1[0] - p_vertex[0], p1[1] - p_vertex[1]])
    v2 = np.array([p3[0] - p_vertex[0], p3[1] - p_vertex[1]])
    
    # Normas
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return float('nan')
        
    # Produto escalar
    dot_product = np.dot(v1, v2)
    
    # Cosseno do ângulo
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Correção de erro numérico para manter no intervalo [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta)
    return math.degrees(angle_rad)

def ponto_em_bbox(px: float, py: float, bbox: Bbox, margem_px: float = 0.0) -> bool:
    """
    ponto_em_bbox: Verifica se um ponto (px, py) está dentro de um bbox, considerando uma margem de tolerância.

    Args:
        px (float): Coordenada X do ponto.
        py (float): Coordenada Y do ponto.
        bbox (Bbox): Dicionário com chaves x_min, y_min, x_max, y_max.
        margem_px (float, optional): Margem de tolerância em pixels para expandir o bbox. Padrão é 0.0.

    Returns:
        bool: True se o ponto estiver contido na área expandida do bbox.
    """
    x_min = bbox["x_min"] - margem_px
    y_min = bbox["y_min"] - margem_px
    x_max = bbox["x_max"] + margem_px
    y_max = bbox["y_max"] + margem_px
    
    return (x_min <= px <= x_max) and (y_min <= py <= y_max)

def calcular_area_bbox(bbox: Bbox) -> float:
    """
    calcular_area_bbox: Calcula a área de um bbox.

    Args:
        bbox (Bbox): Dicionário com chaves x_min, y_min, x_max, y_max.

    Returns:
        float: Área do bbox (width * height).
    """
    width = max(0.0, bbox["x_max"] - bbox["x_min"])
    height = max(0.0, bbox["y_max"] - bbox["y_min"])
    return width * height
