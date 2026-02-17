import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..util.io_arquivos import ler_json
from ..util.contratos import InstanciaVacaAnotada, ImagemAnotada, KeypointPadronizado, Bbox, NomeKeypoint
from ..util.geometria import ponto_em_bbox, calcular_area_bbox

# Mapeamento do nome no Label Studio para o nome interno padronizado
MAPA_LABEL_STUDIO = {
    "Withers": "withers",
    "Back": "back",
    "Hook_up": "hook_up",
    "Hook_Up": "hook_up",
    "Hook up": "hook_up",
    "Hook_down": "hook_down",
    "Hook_Down": "hook_down",
    "Hook down": "hook_down",
    "Hip": "hip",
    "Tail_head": "tail_head",
    "Tail_Head": "tail_head",
    "Tail head": "tail_head",
    "Pin_up": "pin_up",
    "Pin_Up": "pin_up",
    "Pin up": "pin_up",
    "Pin_down": "pin_down",
    "Pin_Down": "pin_down",
    "Pin down": "pin_down"
}

LISTA_KEYPOINTS_ORDENADA: List[NomeKeypoint] = [
    "withers", "back", "hook_up", "hook_down", 
    "hip", "tail_head", "pin_up", "pin_down"
]

def carregar_anotacoes_label_studio(caminho_dataset_root: Path, logger: logging.Logger) -> List[ImagemAnotada]:
    """
    carregar_anotacoes_label_studio: Lê recursivamente arquivos JSON do Label Studio e estrutura os dados.

    Percorre o diretório `caminho_dataset_root`, lê os arquivos JSON,
    identifica a imagem correspondente e processa as anotações.

    Args:
        caminho_dataset_root (Path): Diretório raiz contendo os arquivos JSON e as imagens.
        logger (logging.Logger): Logger.

    Returns:
        List[ImagemAnotada]: Lista de imagens anotadas com seus caminhos, dimensões e instâncias.
    """
    imagens_anotadas: List[ImagemAnotada] = []
    arquivos_json = list(caminho_dataset_root.glob("**/*.json"))
    
    logger.info(f"Encontrados {len(arquivos_json)} arquivos JSON de anotação em {caminho_dataset_root}")

    for arq in arquivos_json:
        try:
            dados_json = ler_json(arq)
            tasks = dados_json if isinstance(dados_json, list) else [dados_json]

            for task in tasks:
                img_anotada = _processar_task_label_studio(task, arq, caminho_dataset_root, logger)
                if img_anotada:
                    imagens_anotadas.append(img_anotada)

        except Exception as e:
            logger.error(f"Erro ao processar arquivo {arq}: {e}")

    logger.info(f"Total de imagens carregadas com sucesso: {len(imagens_anotadas)}")
    return imagens_anotadas

def _processar_task_label_studio(task: Dict[str, Any], arquivo_json: Path, root_dir: Path, logger: logging.Logger) -> Optional[ImagemAnotada]:
    """
    _processar_task_label_studio: Processa uma única task (imagem) do Label Studio.

    Analisa o JSON de uma tarefa, tenta localizar o arquivo de imagem correspondente (usando heurísticas de nome e caminho)
    e extrai as anotações de bbox e keypoints.

    Args:
        task (Dict[str, Any]): Dicionário contendo os dados da task (do JSON).
        arquivo_json (Path): Caminho do arquivo JSON sendo processado (usado para resolver caminhos relativos).
        root_dir (Path): Diretório raiz do dataset (para buscas recursivas de imagem).
        logger (logging.Logger): Logger.

    Returns:
        Optional[ImagemAnotada]: Objeto contendo dados da imagem e instâncias, ou None se falhar/não tiver anotações.
    """
    if "annotations" not in task or not task["annotations"]:
        return None
        
    result_list = task["annotations"][0].get("result", [])
    if not result_list:
        return None

    # Tentar descobrir o caminho da imagem
    # O campo 'data' -> 'image' geralmente tem algo como "/data/local-files/?d=dados/raw/..." ou apenas o nome
    # No dataset fornecido, estrutura: anotador/imagens... e anotador/Key_points/1.json
    # O JSON 1.json provavelmente se refere a uma imagem na pasta pai.
    
    # Estratégia heurística para encontrar a imagem:
    # 1. Tentar pelo nome do arquivo no campo 'data.image'
    # 2. Se falhar, tentar procurar imagem com mesmo nome basename do JSON mas nas extensões jpg/png
    
    nome_imagem_referencia = ""
    if "data" in task and "image" in task["data"]:
        nome_imagem_referencia = task["data"]["image"]
        # Limpar prefixos comuns do LS se houver (ex: hostname, etc)
        nome_imagem_referencia = Path(nome_imagem_referencia).name
    
    # Procurar imagem no diretório pai do JSON ou no avô
    # JSON: .../Key_points/1.json -> Imagem: .../1.jpg ?? Ou .../IMG_123.jpg?
    # Se o nome no data.image for util, usamos ele.
    
    candidatos = []
    
    # Busca 1: Pelo nome exato referenciado no JSON (se existir), procurando recursivamente no root ou relativo
    if nome_imagem_referencia:
        candidatos.append(arquivo_json.parent.parent / nome_imagem_referencia) # ../nome.jpg
        candidatos.append(arquivo_json.parent / nome_imagem_referencia) # ./nome.jpg

    # Busca 2: Pelo stem do JSON (ex: 1.json -> 1.jpg)
    stem = arquivo_json.stem
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    
    for ext in extensions:
        candidatos.append(arquivo_json.parent.parent / (stem + ext))
        candidatos.append(arquivo_json.parent / (stem + ext))
    
    caminho_imagem_final = None
    for cand in candidatos:
        if cand.exists():
            caminho_imagem_final = cand
            break
            
    if not caminho_imagem_final:
        # Tentar busca recursiva pelo nome da imagem se tivermos o nome
        if nome_imagem_referencia:
            encontrados = list(root_dir.rglob(nome_imagem_referencia))
            if encontrados:
                caminho_imagem_final = encontrados[0]

    if not caminho_imagem_final:
        logger.warning(f"Imagem não encontrada para JSON {arquivo_json}. Ref: {nome_imagem_referencia}")
        return None

    # Extrair Dados
    bboxes = []
    keypoints = []

    img_width = 0
    img_height = 0

    for item in result_list:
        tipo = item.get("type")
        val = item.get("value", {})
        
        if img_width == 0 and "original_width" in item:
            img_width = item["original_width"]
            img_height = item["original_height"]

        if tipo == "rectanglelabels":
            if "cow" in val.get("rectanglelabels", []):
                bbox = {
                    "id": item.get("id"),
                    "x_pct": val["x"], "y_pct": val["y"], 
                    "w_pct": val["width"], "h_pct": val["height"],
                    "original_width": item.get("original_width"),
                    "original_height": item.get("original_height")
                }
                bboxes.append(bbox)

        elif tipo == "keypointlabels":
            labels = val.get("keypointlabels", [])
            if labels:
                label_norm = _normalizar_nome_keypoint(labels[0])
                if label_norm:
                    kp = {
                        "id": item.get("id"),
                        "x_pct": val["x"], "y_pct": val["y"],
                        "label": label_norm,
                        "original_width": item.get("original_width"),
                        "original_height": item.get("original_height")
                    }
                    keypoints.append(kp)

    if not bboxes:
         return None

    if img_width == 0 or img_height == 0:
        logger.warning(f"Dimensões zeradas/ausentes em {arquivo_json}. Ignorando.")
        return None

    instancias = []
    
    for bbox_raw in bboxes:
        W = float(bbox_raw["original_width"])
        H = float(bbox_raw["original_height"])
        
        x_min = (float(bbox_raw["x_pct"]) / 100.0) * W
        y_min = (float(bbox_raw["y_pct"]) / 100.0) * H
        width = (float(bbox_raw["w_pct"]) / 100.0) * W
        height = (float(bbox_raw["h_pct"]) / 100.0) * H
        x_max = x_min + width
        y_max = y_min + height
        
        bbox_struct: Bbox = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        }
        
        instancias.append({
            "bbox": bbox_struct,
            "kps_candidatos": [],
            "area": calcular_area_bbox(bbox_struct)
        })

    for kp in keypoints:
        W = float(kp["original_width"])
        H = float(kp["original_height"])
        kp_x_px = (float(kp["x_pct"]) / 100.0) * W
        kp_y_px = (float(kp["y_pct"]) / 100.0) * H
        
        candidatos = []
        for i, inst in enumerate(instancias):
            if ponto_em_bbox(kp_x_px, kp_y_px, inst["bbox"], margem_px=5.0):
                candidatos.append(i)
        
        if not candidatos:
            continue
        
        melhor_idx = -1
        menor_area = float('inf')
        for idx in candidatos:
            area = instancias[idx]["area"]
            if area < menor_area:
                menor_area = area
                melhor_idx = idx
        
        instancias[melhor_idx]["kps_candidatos"].append({
            "label": kp["label"],
            "x_px": kp_x_px,
            "y_px": kp_y_px,
            "v": 2
        })

    instancias_final: List[InstanciaVacaAnotada] = []
    for inst in instancias:
        kps_finais: List[KeypointPadronizado] = []
        mapa_kps = {k["label"]: k for k in inst["kps_candidatos"]}
        
        for nome_ref in LISTA_KEYPOINTS_ORDENADA:
            if nome_ref in mapa_kps:
                k = mapa_kps[nome_ref]
                kps_finais.append({
                    "label": nome_ref,
                    "x_px": k["x_px"],
                    "y_px": k["y_px"],
                    "v": 2
                })
            else:
                kps_finais.append({
                    "label": nome_ref,
                    "x_px": 0.0,
                    "y_px": 0.0,
                    "v": 0
                })
        
        instancias_final.append({
            "bbox": inst["bbox"],
            "keypoints": kps_finais
        })

    return {
        "caminho_imagem": caminho_imagem_final,
        "largura": int(img_width),
        "altura": int(img_height),
        "instancias": instancias_final
    }

def _normalizar_nome_keypoint(raw: str) -> Optional[NomeKeypoint]:
    """
    _normalizar_nome_keypoint: Normaliza o nome do keypoint vindo do Label Studio.

    Mapeia variações de nomes (ex: "Hook up", "Hook_Up") para o identificador interno padronizado
    definido em contratos.py. Utiliza um mapa de conversão explícito e estratégias de normalização
    de string (lowercase, replace).

    Args:
        raw (str): Nome cru do keypoint vindo do Label Studio.

    Returns:
        Optional[NomeKeypoint]: Nome normalizado ou None se não reconhecido.
    """
    if raw in MAPA_LABEL_STUDIO:
        return MAPA_LABEL_STUDIO[raw]
    norm = raw.strip().replace(" ", "_").lower()
    if norm in LISTA_KEYPOINTS_ORDENADA:
        return norm # type: ignore
    return None
