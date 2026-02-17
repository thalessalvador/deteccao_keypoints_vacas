import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..util.io_arquivos import ler_json
from ..util.contratos import InstanciaVacaAnotada, ImagemAnotada, KeypointPadronizado, Bbox, NomeKeypoint, LISTA_KEYPOINTS_ORDENADA
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

def carregar_anotacoes_label_studio(caminho_dataset_root: Path, logger: logging.Logger) -> List[ImagemAnotada]:
    """
    carregar_anotacoes_label_studio: Processa arquivos JSON do Label Studio exportados em formato 'Raw' ou 'JSON-MIN'.
    
    Ajustado para estrutura: raw/dataset_keypoints/[anotador]/Key_points/[id_sem_extensao]
    
    Args:
        caminho_dataset_root (Path): Diretório raiz contendo as pastas dos anotadores.
        logger (logging.Logger): Logger.

    Returns:
        List[ImagemAnotada]: Lista de objetos contendo caminho da imagem e suas anotações.
    """
    contagem_arquivos = 0
    imagens_anotadas: List[ImagemAnotada] = []
    
    # Busca arquivos dentro de pastas chamadas "Key_points"
    # Como não temos extensão .json garantida, vamos listar tudo dentro de Key_points
    padrao_busca = "**/Key_points/*"
    arquivos_candidatos = list(caminho_dataset_root.glob(padrao_busca))
    
    # Se não achar nada com Key_points, tenta busca genérica .json (caso mude estrutura)
    if not arquivos_candidatos:
        arquivos_candidatos = list(caminho_dataset_root.rglob("*.json"))
        
    arquivos_validos = [f for f in arquivos_candidatos if f.is_file()]
    logger.info(f"Encontrados {len(arquivos_validos)} arquivos candidatos em {caminho_dataset_root}")

    for arq in arquivos_validos:
        try:
            dados = ler_json(arq)
            contagem_arquivos += 1
            
            # Normalizar para lista de tasks/anotações
            # O formato identificado é um objeto único com "task" e "result" na raiz
            lista_para_processar = []
            if isinstance(dados, dict):
                lista_para_processar.append(dados)
            elif isinstance(dados, list):
                lista_para_processar.extend(dados)
                
            for item in lista_para_processar:
                img_anotada = _processar_item_label_studio(item, arq, caminho_dataset_root, logger)
                if img_anotada:
                    # Contagem para log de resumo
                    n_vacas = len(img_anotada["instancias"])
                    n_kps_total = 0
                    for inst in img_anotada["instancias"]:
                        # Conta apenas KPs válidos (v > 0)
                        n_kps_inst = sum(1 for k in inst["keypoints"] if k["v"] > 0)
                        n_kps_total += n_kps_inst
                    
                    logger.info(f"Processado: {img_anotada['caminho_imagem'].name} | Vacas: {n_vacas} | Keypoints: {n_kps_total}")
                    
                    imagens_anotadas.append(img_anotada)
                    
        except Exception as e:
            # Logger debug para não poluir se for arquivo de sistema
            logger.debug(f"Falha ao ler/processar arquivo {arq}: {e}")
            continue

    logger.info(f"Total de imagens carregadas com sucesso: {len(imagens_anotadas)}")
    return imagens_anotadas

def _processar_item_label_studio(item: Dict[str, Any], arquivo_json: Path, root_dir: Path, logger: logging.Logger) -> Optional[ImagemAnotada]:
    """
    Adaptação para processar o formato específico onde 'result' está na raiz e 'task' contém metadados.
    """
    # Identificar onde estão os resultados e o caminho da imagem
    results = []
    img_path_raw = ""
    
    # Caso 1: Formato observado (raiz com 'result' e 'task')
    if "result" in item and "task" in item:
        results = item["result"]
        img_data = item["task"].get("data", {})
        img_path_raw = img_data.get("img") or img_data.get("image") or ""
        
    # Caso 2: Formato tradicional de export (lista de tasks)
    elif "annotations" in item:
        if item["annotations"]:
            results = item["annotations"][0].get("result", [])
        img_data = item.get("data", {})
        img_path_raw = img_data.get("img") or img_data.get("image") or ""
        
    if not results:
        return None
        
    # Extrair nome da imagem
    nome_imagem_referencia = ""
    if img_path_raw:
        # Decodificar URL se necessário (ex: %20 -> espaço)
        import urllib.parse
        decoded_path = urllib.parse.unquote(img_path_raw)
        # Pegar apenas o nome do arquivo, ignorando caminhos falsos do LS (/data/local-files/...)
        nome_imagem_referencia = Path(decoded_path).name
    
    # Buscar imagem no disco
    caminho_imagem_final = None
    
    # 1. Tentar na pasta pai do diretório Key_points (estrutura: Anotador/Key_points/1 -> Anotador/img.jpg)
    if "Key_points" in arquivo_json.parent.name:
         dir_anotador = arquivo_json.parent.parent
         # Se tivermos o nome da referência
         if nome_imagem_referencia:
             candidato = dir_anotador / nome_imagem_referencia
             if candidato.exists():
                 caminho_imagem_final = candidato
         
         # 2. Se falhar, tentar buscar pela raiz do anotador qualquer arquivo com esse nome
         if not caminho_imagem_final and nome_imagem_referencia:
             encontrados = list(dir_anotador.glob(nome_imagem_referencia))
             if encontrados:
                 caminho_imagem_final = encontrados[0]
                 
    # 3. Busca global se ainda não achou
    if not caminho_imagem_final and nome_imagem_referencia:
        encontrados = list(root_dir.rglob(nome_imagem_referencia))
        if encontrados:
            caminho_imagem_final = encontrados[0]

    if not caminho_imagem_final:
        # Se não achou a imagem, não podemos usar
        # logger.warning(f"Imagem não encontrada para JSON {arquivo_json.name}. Ref: {nome_imagem_referencia}")
        return None

    # Extrair Dados
    bboxes = []
    keypoints = []

    img_width = 0
    img_height = 0

    for res in results:
        tipo = res.get("type")
        val = res.get("value", {})
        
        if img_width == 0 and "original_width" in res:
            img_width = res["original_width"]
            img_height = res["original_height"]

        if tipo == "rectanglelabels":
            if "cow" in val.get("rectanglelabels", []):
                bbox = {
                    "id": res.get("id"),
                    "x_pct": val["x"], "y_pct": val["y"], 
                    "w_pct": val["width"], "h_pct": val["height"],
                    "original_width": res.get("original_width"),
                    "original_height": res.get("original_height")
                }
                bboxes.append(bbox)

        elif tipo == "keypointlabels":
            labels = val.get("keypointlabels", [])
            if labels:
                label_norm = _normalizar_nome_keypoint(labels[0])
                if label_norm:
                    kp = {
                        "id": res.get("id"),
                        "x_pct": val["x"], "y_pct": val["y"],
                        "label": label_norm,
                        "original_width": res.get("original_width"),
                        "original_height": res.get("original_height")
                    }
                    keypoints.append(kp)

    if not bboxes:
         return None

    if img_width == 0 or img_height == 0:
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
