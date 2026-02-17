import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from .contratos import RespostaInferenciaPose, InstanciaVacaAnotada

# Cores (BGR)
COLOR_BBOX = (0, 255, 0)      # Verde
COLOR_KPT_VIS = (0, 0, 255)   # Vermelho
COLOR_KPT_HID = (0, 255, 255) # Amarelo (se quisermos diferenciar, mas spec diz v=2 visivel)
COLOR_SKELETON = (255, 100, 0)# Azulado

# Conexões para desenhar esqueleto (opcional para visualização)
# withers(1), back(2), hook_up(3), hook_down(4), hip(5), tail_head(6), pin_up(7), pin_down(8)
# Índices internos (0-based da lista ordenada):
# 0:withers, 1:back, 2:hook_up, 3:hook_down, 4:hip, 5:tail_head, 6:pin_up, 7:pin_down
SKELETON_CONNECTIONS = [
    (0, 1), # withers -> back
    (1, 4), # back -> hip (ou ligar em hooks?)
    (2, 4), # hook_up -> hip
    (3, 4), # hook_down -> hip
    (4, 5), # hip -> tail_head
    (4, 6), # hip -> pin_up
    (4, 7), # hip -> pin_down
    (6, 7), # pin_up -> pin_down
    (2, 3), # hook_up -> hook_down
]

def plotar_keypoints_na_imagem(
    caminho_imagem: Path, 
    instancias: List[InstanciaVacaAnotada], 
    caminho_saida: Path, 
    logger: logging.Logger
) -> Optional[Path]:
    """
    plotar_keypoints_na_imagem: Desenha bboxes e keypoints na imagem e salva.

    Args:
        caminho_imagem (Path): Caminho da imagem original.
        instancias (List[InstanciaVacaAnotada]): Lista de instâncias com bboxes e keypoints.
        caminho_saida (Path): Caminho onde a imagem desenhada será salva.
        logger (logging.Logger): Logger.

    Returns:
        Optional[Path]: Caminho do arquivo salvo ou None em caso de erro.
    """
    if not caminho_imagem.exists():
        logger.error(f"Imagem não encontrada para plot: {caminho_imagem}")
        return None

    try:
        img = cv2.imread(str(caminho_imagem))
        if img is None:
            logger.error(f"Falha ao abrir imagem com cv2: {caminho_imagem}")
            return None

        # Desenhar cada instância
        for inst in instancias:
            # Bbox
            bbox = inst["bbox"]
            x1, y1 = int(bbox["x_min"]), int(bbox["y_min"])
            x2, y2 = int(bbox["x_max"]), int(bbox["y_max"])
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)
            
            # Keypoints
            kps_list = inst["keypoints"] # Lista ordenada
            
            # Desenhar pontos
            for i, kp in enumerate(kps_list):
                if kp["v"] > 0: # Visível ou Oculto (se quisermos plotar oculto)
                    px, py = int(kp["x_px"]), int(kp["y_px"])
                    cv2.circle(img, (px, py), 4, COLOR_KPT_VIS, -1)
                    # Label (opcional, pode poluir)
                    # cv2.putText(img, str(i+1), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Desenhar esqueleto
            for i1, i2 in SKELETON_CONNECTIONS:
                if i1 < len(kps_list) and i2 < len(kps_list):
                    kp1 = kps_list[i1]
                    kp2 = kps_list[i2]
                    
                    if kp1["v"] > 0 and kp2["v"] > 0:
                        p1 = (int(kp1["x_px"]), int(kp1["y_px"]))
                        p2 = (int(kp2["x_px"]), int(kp2["y_px"]))
                        cv2.line(img, p1, p2, COLOR_SKELETON, 2)

        # Salvar
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(caminho_saida), img)
        return caminho_saida

    except Exception as e:
        logger.error(f"Erro ao plotar imagem {caminho_imagem}: {e}")
        return None
