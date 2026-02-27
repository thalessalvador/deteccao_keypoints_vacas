import math
import numpy as np
from typing import Tuple, List
from .contratos import Bbox


def calcular_distancia(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    calcular_distancia: Calcula a distancia euclidiana entre dois pontos (x, y).

    Args:
        p1 (Tuple[float, float]): Coordenadas (x, y) do ponto 1.
        p2 (Tuple[float, float]): Coordenadas (x, y) do ponto 2.

    Returns:
        float: Distancia euclidiana entre p1 e p2.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calcular_angulo(p1: Tuple[float, float], p_vertex: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    calcular_angulo: Calcula o angulo (em graus) formado por tres pontos (A, V, B), com vertice em V.

    O angulo retornado esta no intervalo [0, 180].

    Args:
        p1 (Tuple[float, float]): Ponto A.
        p_vertex (Tuple[float, float]): Ponto vertice V.
        p3 (Tuple[float, float]): Ponto B.

    Returns:
        float: Angulo em graus. Retorna NaN se algum vetor tiver norma zero.
    """
    v1 = np.array([p1[0] - p_vertex[0], p1[1] - p_vertex[1]])
    v2 = np.array([p3[0] - p_vertex[0], p3[1] - p_vertex[1]])

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return float('nan')

    dot_product = np.dot(v1, v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    return math.degrees(angle_rad)


def ponto_em_bbox(px: float, py: float, bbox: Bbox, margem_px: float = 0.0) -> bool:
    """
    ponto_em_bbox: Verifica se um ponto (px, py) esta dentro de um bbox, com margem opcional.

    Args:
        px (float): Coordenada X do ponto.
        py (float): Coordenada Y do ponto.
        bbox (Bbox): Bounding box com chaves x_min, y_min, x_max, y_max.
        margem_px (float, optional): Margem de tolerancia em pixels para expandir o bbox.

    Returns:
        bool: True se o ponto estiver dentro da area expandida do bbox.
    """
    x_min = bbox["x_min"] - margem_px
    y_min = bbox["y_min"] - margem_px
    x_max = bbox["x_max"] + margem_px
    y_max = bbox["y_max"] + margem_px

    return (x_min <= px <= x_max) and (y_min <= py <= y_max)


def calcular_area_bbox(bbox: Bbox) -> float:
    """
    calcular_area_bbox: Calcula a area de um bounding box.

    Args:
        bbox (Bbox): Bounding box com chaves x_min, y_min, x_max, y_max.

    Returns:
        float: Area do bbox (largura * altura), truncando valores negativos para 0.
    """
    width = max(0.0, bbox["x_max"] - bbox["x_min"])
    height = max(0.0, bbox["y_max"] - bbox["y_min"])
    return width * height


def calcular_area_poligono(vertices: List[Tuple[float, float]]) -> float:
    """
    calcular_area_poligono: Calcula a area de um poligono por vertices ordenados.
    
    Utiliza a formula de Shoelace para computar a area no plano cartesiano.
    
    Args:
        vertices (List[Tuple[float, float]]): Lista ordenada de vertices (x, y).
    
    Returns:
        float: Area do poligono. Retorna 0.0 quando houver menos de 3 vertices.
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return abs(area) / 2.0


def calcular_distancia_ponto_reta(p: Tuple[float, float], r1: Tuple[float, float], r2: Tuple[float, float]) -> float:
    """
    calcular_distancia_ponto_reta: Calcula a distancia perpendicular de um ponto a uma reta.
    
    A reta e definida pelos pontos r1 e r2. Se r1 e r2 coincidirem, a funcao retorna
    a distancia entre p e r1.
    
    Args:
        p (Tuple[float, float]): Ponto de interesse.
        r1 (Tuple[float, float]): Primeiro ponto da reta.
        r2 (Tuple[float, float]): Segundo ponto da reta.
    
    Returns:
        float: Distancia perpendicular de p a reta (r1, r2) ou distancia ponto-a-ponto no caso degenerado.
    """
    if r1 == r2:
        return calcular_distancia(p, r1)

    num = abs((r2[1] - r1[1]) * p[0] - (r2[0] - r1[0]) * p[1] + r2[0] * r1[1] - r2[1] * r1[0])
    den = math.sqrt((r2[1] - r1[1])**2 + (r2[0] - r1[0])**2)

    return num / den
