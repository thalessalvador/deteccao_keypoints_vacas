import logging
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Any

from ..util.contratos import ImagemAnotada, InstanciaVacaAnotada, LISTA_KEYPOINTS_ORDENADA
from ..util.io_arquivos import garantir_diretorio

def converter_para_yolo_pose(imagens_anotadas: List[ImagemAnotada], dir_saida: Path, logger: logging.Logger) -> None:
    """
    converter_para_yolo_pose: Converte dataset interno para formato YOLOv8 Pose.

    Para cada imagem anotada:
    1. Copia a imagem para dir_saida/images
    2. Cria arquivo .txt correspondente em dir_saida/labels
    3. Escreve cada instância como uma linha no formato YOLO Pose:
       <class> <x> <y> <w> <h> <kp1_x> <kp1_y> <kp1_v> ...

    Args:
        imagens_anotadas (List[ImagemAnotada]): Lista de imagens anotadas.
        dir_saida (Path): Diretório base para saída (ex: dados/processados/yolo_pose).
        logger (logging.Logger): Logger.

    Returns:
        None
    """
    dir_images = dir_saida / "images"
    dir_labels = dir_saida / "labels"
    
    garantir_diretorio(dir_images)
    garantir_diretorio(dir_labels)
    
    count_sucesso = 0
    
    for img_data in imagens_anotadas:
        try:
            # 1. Copiar Imagem
            src_img = img_data["caminho_imagem"]
            if not src_img.exists():
                logger.warning(f"Imagem original não encontrada: {src_img}. Pulando.")
                continue
                
            dst_img = dir_images / src_img.name
            shutil.copy2(src_img, dst_img)
            
            # 2. Criar Label TXT
            label_file = dir_labels / (src_img.stem + ".txt")
            
            linhas_txt = []
            W_img = float(img_data["largura"])
            H_img = float(img_data["altura"])
            
            for inst in img_data["instancias"]:
                linha = _formatar_linha_yolo(inst, W_img, H_img)
                linhas_txt.append(linha)
                
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(linhas_txt))
                
            count_sucesso += 1
            
        except Exception as e:
            logger.error(f"Erro ao converter imagem {img_data.get('caminho_imagem')}: {e}")
            
    logger.info(f"Conversão concluída. {count_sucesso}/{len(imagens_anotadas)} imagens processadas.")

def _formatar_linha_yolo(inst: InstanciaVacaAnotada, img_w: float, img_h: float) -> str:
    """
    _formatar_linha_yolo: Formata os dados de uma instância (vaca) para uma string de linha no formato YOLO Pose.

    Realiza a normalização de coordenadas (bbox e keypoints) pelo tamanho da imagem e formata a string
    de acordo com a especificação YOLO: class_id cx cy w h k1x k1y k1v ...

    Args:
        inst (InstanciaVacaAnotada): Dados da instância (bbox + keypoints).
        img_w (float): Largura da imagem original.
        img_h (float): Altura da imagem original.

    Returns:
        str: String formatada pronta para ser escrita no arquivo .txt.
    """
    bbox = inst["bbox"]
    
    # Centro e WH normalizados
    box_w = bbox["x_max"] - bbox["x_min"]
    box_h = bbox["y_max"] - bbox["y_min"]
    box_cx = bbox["x_min"] + (box_w / 2.0)
    box_cy = bbox["y_min"] + (box_h / 2.0)
    
    # Normalizar
    ncx = box_cx / img_w
    ncy = box_cy / img_h
    nw = box_w / img_w
    nh = box_h / img_h
    
    # YOLO Class ID fixo = 0 (cow)
    # Formato: 0 ncx ncy nw nh k1x k1y k1v k2x k2y k2v ...
    parts = [0, f"{ncx:.6f}", f"{ncy:.6f}", f"{nw:.6f}", f"{nh:.6f}"]
    
    for kp in inst["keypoints"]:
        # Se v=0 (inexistente), x=0 y=0. Se v=2 (visivel) normaliza.
        # YOLO specs: x,y normalized. v: 0=missing/not labeled, 1=invisible, 2=visible.
        if kp["v"] == 0:
             parts.extend(["0.000000", "0.000000", "0"])
        else:
            nkx = kp["x_px"] / img_w
            nky = kp["y_px"] / img_h
            parts.extend([f"{nkx:.6f}", f"{nky:.6f}", "2"]) # Forçando 2 se existir na lista do parser (que já filtra visible)
            
    return " ".join(str(p) for p in parts)

def gerar_dataset_yaml_ultralytics(dir_saida: Path, logger: logging.Logger) -> Path:
    """
    gerar_dataset_yaml_ultralytics: Cria o arquivo dataset.yaml necessário para treino.

    Como estamos usando Cross Validation ou split posterior, este YAML pode apontar apenas
    para o diretório de imagens local, ou podemos não usar se o script de treino criar 
    seus próprios YAMLs por fold.
    
    Aqui criaremos um YAML genérico "base" apontando para o diretório de imagens.

    Args:
        dir_saida (Path): Diretório raiz do dataset YOLO.
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho do arquivo yaml criado.
    """
    yaml_path = dir_saida / "dataset.yaml"
    
    # Ponto importante: YOLO exige train/val paths.
    # Se vamos fazer K-Fold no script de treino, o script de treino vai gerar os yamls específicos.
    # Mas é bom ter um 'all.yaml' apontando tudo para train e val.
    
    # Atenção: Caminhos absolutos são mais seguros.
    abs_path = dir_saida.resolve()
    
    config = {
        "path": str(abs_path),
        "train": "images", # Usa todas imagens como treino base (o split deve ser feito via argumento ou files)
        "val": "images",   # Mesmo dir
        "#kpt_shape": "[8, 3]", # Comentário informativo
        "kpt_shape": [8, 3],
        "names": {
            0: "cow"
        },
        # Adicionando nomes dos keypoints para referência explícita da ordem
        "kpt_names": LISTA_KEYPOINTS_ORDENADA
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
        
    logger.info(f"Arquivo dataset.yaml criado em {yaml_path}")
    return yaml_path
