from pathlib import Path
from typing import List, Dict, Any
import logging
from .contratos import NomeKeypoint

def validar_arquivo_existe(caminho: Path, logger: logging.Logger) -> bool:
    """
    validar_arquivo_existe: Verifica a existência de um arquivo e loga erro se não encontrar.

    Args:
        caminho (Path): Caminho do arquivo a ser verificado.
        logger (logging.Logger): Logger para registrar erro caso o arquivo não exista.

    Returns:
        bool: True se o arquivo existe, False caso contrário.
    """
    if not caminho.exists():
        logger.error(f"Arquivo não encontrado: {caminho}")
        return False
    return True

def validar_dataset_yolo_pose(dir_yolo: Path, labels_cfg: Dict[str, Any], logger: logging.Logger) -> dict:
    """
    validar_dataset_yolo_pose: Realiza validação básica da integridade de um dataset no formato YOLO Pose.

    Verifica se as pastas 'images' e 'labels' existem, conta o número de arquivos
    e alerta sobre desbalanceamento entre imagens e labels.

    Args:
        dir_yolo (Path): Caminho raiz do dataset YOLO (deve conter pastas 'images' e 'labels').
        labels_cfg (Dict[str, Any]): Configuração dos labels (não utilizado na validação estrutural básica, mas mantido para extensibilidade).
        logger (logging.Logger): Logger para registrar avisos e erros.

    Returns:
        dict: Dicionário contendo status ('ok' ou 'erro'), métricas (total_imagens, total_labels) e motivo em caso de erro.
    """
    images_dir = dir_yolo / "images"
    labels_dir = dir_yolo / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"Estrutura do dataset inválida: {dir_yolo}")
        return {"status": "erro", "motivo": "diretorios_ausentes"}

    total_imagens = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    total_labels = len(list(labels_dir.glob("*.txt")))

    if total_imagens == 0:
         logger.warning("Dataset de imagens vazio.")
    
    if total_imagens != total_labels:
        logger.warning(f"Desbalanceamento: {total_imagens} imagens vs {total_labels} labels.")

    return {
        "status": "ok",
        "total_imagens": total_imagens,
        "total_labels": total_labels
    }
