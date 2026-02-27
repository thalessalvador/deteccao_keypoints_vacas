import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from ultralytics import YOLO

from ..util.contratos import RespostaInferenciaPose, InstanciaVacaAnotada, Bbox, KeypointPadronizado, NomeKeypoint
from ..util.visualizacao import plotar_keypoints_na_imagem, plotar_preview_augmentacao_keypoints
from ..util.geometria import calcular_area_bbox
from ..classificacao.gerador_dataset_features import _selecionar_instancia_alvo, _ler_cfg_augmentacao_keypoints, _gerar_keypoints_com_ruido

LISTA_KEYPOINTS_ORDENADA_YOLO: List[NomeKeypoint] = [
    "withers", "back", "hook_up", "hook_down", 
    "hip", "tail_head", "pin_up", "pin_down"
]

def inferir_keypoints_em_imagem(
    caminho_imagem: Path, 
    caminho_modelo: Path, 
    config: Dict[str, Any], 
    desenhar: bool, 
    dir_saida_plot: Optional[Path],
    logger: logging.Logger,
    desenhar_augmentacao: bool = False,
    aug_copias: Optional[int] = None,
) -> RespostaInferenciaPose:
    """
    inferir_keypoints_em_imagem: Realiza inferência de pose em uma imagem única.

    Args:
        caminho_imagem (Path): Caminho da imagem de entrada.
        caminho_modelo (Path): Caminho do arquivo de pesos (.pt).
        config (Dict[str, Any]): Configuração (para thresholds, device, etc).
        desenhar (bool): Se deve gerar imagem com plot.
        dir_saida_plot (Optional[Path]): Diretório para salvar o plot (se desenhar=True).
        logger (logging.Logger): Logger.
        desenhar_augmentacao (bool): Se deve gerar preview de keypoints augmentados por ruido.
        aug_copias (Optional[int]): Numero de copias augmentadas no preview. Se None, usa config.

    Returns:
        RespostaInferenciaPose: Dicionário contendo as instâncias detectadas e metadados.
    """
    if not caminho_imagem.exists():
        logger.error(f"Imagem não encontrada: {caminho_imagem}")
        return {"instancias": [], "caminho_imagem": str(caminho_imagem)}

    pose_cfg = config.get("pose", {})
    conf_min = config.get("classificacao", {}).get("selecao_instancia", {}).get("conf_min", 0.4) 
    # Ou usar conf do YOLO. O YOLO tem conf threshold na inferencia.
    
    device = pose_cfg.get("device", "0")
    if device != "cpu":
        if not torch.cuda.is_available():
            logger.warning(f"CUDA não disponível, mas device='{device}' foi solicitado. Forçando device='cpu'.")
            device = "cpu"
    
    try:
        model = YOLO(caminho_modelo)
        
        # Inferencia
        # conf: Confidence threshold
        # iou: NMS IoU threshold
        results = model.predict(source=str(caminho_imagem), save=False, conf=0.25, device=device, verbose=False)
        
        result = results[0] # Primeira imagem (batch 1)
        
        instancias: List[InstanciaVacaAnotada] = []
        
        # Iterar sobre detecções
        if result.boxes and result.keypoints:
            boxes = result.boxes.cpu().numpy()
            keypoints = result.keypoints.cpu().numpy() # shape (N, 8, 3) se xyconf normalized? Não, xy e conf.
            # keypoints.data é tensor. .xy é (N, 8, 2), .conf é (N, 8), .xyn normalized
            # Melhor pegar .data que é (N, 8, 3) [x, y, conf]
            
            kpts_data = result.keypoints.data.cpu().numpy() # (N, num_kpts, 3)
            
            for i, box in enumerate(boxes):
                # Bbox
                # box.xyxy[0] -> [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0]
                conf_bbox = box.conf[0]
                
                # Filtrar por confiança (opcional, já filtramos no predict com conf=0.25 mas podemos refinar)
                
                bbox_struct: Bbox = {
                    "x_min": float(x1),
                    "y_min": float(y1),
                    "x_max": float(x2),
                    "y_max": float(y2)
                }
                
                # Keypoints
                kps_instancia: List[KeypointPadronizado] = []
                # kpts_data[i] é (8, 3) -> [x, y, conf]
                
                kpts_vals = kpts_data[i] # shape (8, 3)
                
                for k_idx, k_val in enumerate(kpts_vals):
                    kx, ky, kconf = k_val
                    
                    label = LISTA_KEYPOINTS_ORDENADA_YOLO[k_idx] if k_idx < len(LISTA_KEYPOINTS_ORDENADA_YOLO) else f"k{k_idx}"
                    
                    # Visibilidade: YOLO retorna conf. Definir threshold de visibilidade?
                    # O dataset YOLO tem v=2 visivel. A rede preve conf.
                    # Se conf < 0.5, considerar oculto ou inexistente?
                    # Para downstream tasks, melhor manter e diferenciar pela conf se precisar.
                    # Contrato pede int v (0, 1, 2).
                    # Heuristica: conf > 0.5 -> 2, conf > 0.1 -> 1, else 0?
                    visibilidade = 2 if kconf > 0.5 else (1 if kconf > 0.1 else 0)
                    
                    kps_instancia.append({
                        "label": label, # type: ignore
                        "x_px": float(kx),
                        "y_px": float(ky),
                        "v": visibilidade
                    })
                
                instancias.append({
                    "bbox": bbox_struct,
                    "keypoints": kps_instancia
                })
        
        resposta: RespostaInferenciaPose = {
            "instancias": instancias,
            "caminho_imagem": str(caminho_imagem)
        }
        
        if desenhar and dir_saida_plot:
            filename = caminho_imagem.name
            caminho_saida = dir_saida_plot / filename
            plotar_keypoints_na_imagem(caminho_imagem, instancias, caminho_saida, logger)

        if desenhar_augmentacao and dir_saida_plot:
            sel_cfg = config.get("classificacao", {}).get("selecao_instancia", {})
            kpts_sel, bbox_sel = _selecionar_instancia_alvo(result, str(caminho_imagem), sel_cfg)
            if kpts_sel is None:
                logger.warning("Preview de augmentacao nao gerado: nenhuma instancia alvo encontrada.")
            else:
                cfg_aug = _ler_cfg_augmentacao_keypoints(config)
                n_copias = int(aug_copias if aug_copias is not None else cfg_aug.get("n_copias", 0))
                if n_copias <= 0:
                    logger.warning("Preview de augmentacao nao gerado: n_copias <= 0.")
                else:
                    img_h, img_w = result.orig_shape
                    kpts_augmentados: List[np.ndarray] = []
                    for _ in range(n_copias):
                        kpts_aug, n_perturbados = _gerar_keypoints_com_ruido(
                            kpts=kpts_sel,
                            bbox_info=bbox_sel,
                            img_w=int(img_w),
                            img_h=int(img_h),
                            cfg_aug=cfg_aug,
                        )
                        if kpts_aug is None or n_perturbados == 0:
                            continue
                        kpts_augmentados.append(kpts_aug)

                    caminho_saida_aug = dir_saida_plot / f"{caminho_imagem.stem}_aug_preview{caminho_imagem.suffix}"
                    salvo = plotar_preview_augmentacao_keypoints(
                        caminho_imagem=caminho_imagem,
                        kpts_original=kpts_sel,
                        kpts_augmentados=kpts_augmentados,
                        caminho_saida=caminho_saida_aug,
                        logger=logger,
                    )
                    if salvo is not None:
                        logger.info(f"Preview de augmentacao salvo em: {salvo}")
                    else:
                        logger.warning("Falha ao salvar preview de augmentacao.")
            
        return resposta

    except Exception as e:
        logger.error(f"Erro na inferência da imagem {caminho_imagem}: {e}")
        return {"instancias": [], "caminho_imagem": str(caminho_imagem)}
