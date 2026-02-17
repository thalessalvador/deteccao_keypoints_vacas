from typing import TypedDict, Literal, List, Optional, Union
from pathlib import Path

# Definição dos nomes dos keypoints
NomeKeypoint = Literal[
    "withers", 
    "back", 
    "hook_up", 
    "hook_down", 
    "hip", 
    "tail_head", 
    "pin_up", 
    "pin_down"
]

LISTA_KEYPOINTS_ORDENADA: List[NomeKeypoint] = [
    "withers", 
    "back", 
    "hook_up", 
    "hook_down", 
    "hip", 
    "tail_head", 
    "pin_up", 
    "pin_down"
]

class KeypointPadronizado(TypedDict):
    """Representação padronizada de um keypoint."""
    label: NomeKeypoint
    x_px: float
    y_px: float
    v: int  # 0: inexistente, 1: oculto, 2: visível

class Bbox(TypedDict):
    """Bounding box [x_min, y_min, x_max, y_max]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class InstanciaVacaAnotada(TypedDict):
    """Dados de uma vaca anotada (bbox + keypoints)."""
    bbox: Bbox
    keypoints: List[KeypointPadronizado] # Deve conter sempre 8 elementos

class ImagemAnotada(TypedDict):
    """Representa uma imagem e suas anotações (múltiplas vacas)."""
    caminho_imagem: Path
    largura: int
    altura: int
    instancias: List[InstanciaVacaAnotada]

class RespostaInferenciaPose(TypedDict):
    """Estrutura de retorno da inferência de pose."""
    instancias: List[InstanciaVacaAnotada]
    caminho_imagem: str
