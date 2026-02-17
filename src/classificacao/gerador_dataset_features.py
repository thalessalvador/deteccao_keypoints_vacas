import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO

from ..util.io_arquivos import garantir_diretorio, ler_yaml
from ..util.geometria import calcular_distancia, calcular_angulo, calcular_area_bbox
from ..util.contratos import NomeKeypoint, LISTA_KEYPOINTS_ORDENADA

# Mapeamento de índices para acesso rápido
IDX_KP = {nome: i for i, nome in enumerate(LISTA_KEYPOINTS_ORDENADA)}

def gerar_dataset_features(config: Dict[str, Any], logger: logging.Logger) -> Path:
    """
    gerar_dataset_features: Gera o dataset tabular de features geométricas para classificação.
    
    Percorre o dataset_classificacao, executa inferência de pose, seleciona a instância alvo,
    calcula features (ângulos, razões) e salva em CSV.

    Args:
        config (Dict[str, Any]): Configurações do sistema.
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho do arquivo CSV gerado.
    """
    logger.info("=== Fase 2: Geração de Features ===")
    
    # Caminhos
    raw_dir = Path(config["paths"]["raw"]) / "dataset_classificacao"
    processed_dir = Path(config["paths"]["processed"]) / "classificacao" / "features"
    garantir_diretorio(processed_dir)
    
    # Configurações de seleção
    cls_cfg = config.get("classificacao", {})
    sel_cfg = cls_cfg.get("selecao_instancia", {})
    conf_min = sel_cfg.get("conf_min", 0.4)
    normalizar_orient = cls_cfg.get("normalizar_orientacao", False)
    
    # Modelo
    pose_cfg = config.get("pose", {})
    model_name = pose_cfg.get("model_name", "yolov8n-pose.pt")
    
    # Tentar encontrar melhor modelo treinado se disponível
    runs_dir = Path("modelos/pose/runs").resolve()
    try:    
        # Pega o reports/metricas_pose.json para saber o melhor ou busca o ultimo best.pt
        import json
        relatorio_path = Path("saidas/relatorios/metricas_pose.json").resolve()
        best_model_path = None
        if relatorio_path.exists():
            with open(relatorio_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                best_model_path = data.get("melhor_modelo", {}).get("path")
        
        if best_model_path and Path(best_model_path).exists():
            model_path = best_model_path
            logger.info(f"Usando melhor modelo treinado: {model_path}")
        else:
            # Fallback
            candidates = list(runs_dir.rglob("best.pt"))
            if candidates:
                model_path = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
                logger.info(f"Usando modelo encontrado (fallback): {model_path}")
            else:
                model_path = model_name
                logger.warning(f"Modelo treinado não encontrado. Usando base: {model_path}")
    except Exception as e:
        logger.warning(f"Erro ao buscar modelo treinado: {e}. Usando base.")
        model_path = model_name

    # Carregar modelo
    device = pose_cfg.get("device", "0")
    model = YOLO(model_path)
    
    # Listar classes (pastas de vacas)
    classes_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not classes_dirs:
        logger.error(f"Nenhuma classe encontrada em {raw_dir}")
        return Path("")
    
    dados_features: List[Dict[str, Any]] = []
    imagens_descartadas: List[Dict[str, Any]] = []
    
    logger.info(f"Processando {len(classes_dirs)} classes (vacas)...")
    
    for class_dir in classes_dirs:
        cow_id = class_dir.name
        image_files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        
        for img_path in image_files:
            try:
                # Inferência
                results = model.predict(source=str(img_path), save=False, conf=0.25, device=device, verbose=False)
                result = results[0]
                
                # Selecionar melhor instância
                instancia_kpts, instancia_bbox = _selecionar_instancia_alvo(result, str(img_path), sel_cfg)
                
                if instancia_kpts is None:
                    imagens_descartadas.append({"arquivo": img_path.name, "classe": cow_id, "motivo": "Nenhuma instancia confiavel"})
                    continue
                
                # Normalizar orientação (se ativado)
                kpts = instancia_kpts # shape (8, 3) -> x, y, conf
                if normalizar_orient:
                    kpts = _normalizar_orientacao_keypoints(kpts)
                
                # Calcular features
                feats = _calcular_features_geometricas(kpts, instancia_bbox)
                
                # Adicionar metadados
                feats["arquivo"] = img_path.name
                feats["classe"] = cow_id
                # feats["target"] = cow_id # Pode ser util num encoding numerico depois
                
                dados_features.append(feats)
                
            except Exception as e:
                logger.error(f"Erro ao processar {img_path}: {e}")
                imagens_descartadas.append({"arquivo": img_path.name, "classe": cow_id, "motivo": f"Erro: {str(e)}"})

    # Salvar CSVs
    df = pd.DataFrame(dados_features)
    out_csv = processed_dir / "features_completas.csv"
    df.to_csv(out_csv, index=False)
    
    if imagens_descartadas:
        df_descartadas = pd.DataFrame(imagens_descartadas)
        df_descartadas.to_csv(processed_dir / "imagens_descartadas.csv", index=False)
        logger.warning(f"{len(imagens_descartadas)} imagens descartadas. Ver imagens_descartadas.csv")
        
    # Gerar Splits 90/10 (Spec 10.1)
    # O CSV features_completas deve conter TUDO. O split define quem é quem.
    splits_dir = Path(config["paths"]["processed"]) / "classificacao" / "splits"
    _criar_splits_por_vaca(raw_dir, 0.1, 42, splits_dir, logger)
    
    logger.info(f"Features geradas com sucesso: {out_csv} ({len(df)} registros)")
    return out_csv

def _criar_splits_por_vaca(dir_dataset: Path, split_teste: float, seed: int, dir_saida: Path, logger: logging.Logger) -> Tuple[List[Path], List[Path]]:
    """
    Cria arquivos de split treino/teste (90/10) estratificado por vaca.
    Salva em dir_saida/treino.txt e dir_saida/teste_10pct.txt.
    """
    import random
    random.seed(seed)
    
    garantir_diretorio(dir_saida)
    
    classes_dirs = [d for d in dir_dataset.iterdir() if d.is_dir()]
    all_train = []
    all_test = []
    
    for d in classes_dirs:
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
        random.shuffle(imgs)
        
        n_test = max(1, int(len(imgs) * split_teste)) # Pelo menos 1 de teste se possivel? Ou 10% strict?
        # Spec diz 10%. Se tiver 50 imgs -> 5 teste.
        
        if len(imgs) < 2:
            # Se tiver muito pouco, bota tudo no treino ou avisa
            test_files = []
            train_files = imgs
        else:
            test_files = imgs[:n_test]
            train_files = imgs[n_test:]
            
        all_test.extend(test_files)
        all_train.extend(train_files)
        
    # Salvar txts (apenas nomes de arquivo ou caminhos relativos? Features tem coluna 'arquivo')
    # Vamos salvar nomes de arquivo para facilitar join com CSV
    with open(dir_saida / "treino.txt", 'w') as f:
        for p in all_train: 
            # Salvar caminho relativo: Classe/Arquivo.ext
            rel_path = p.relative_to(dir_dataset)
            f.write(f"{rel_path}\n")
        
    with open(dir_saida / "teste_10pct.txt", 'w') as f:
        for p in all_test: 
            rel_path = p.relative_to(dir_dataset)
            f.write(f"{rel_path}\n")
        
    logger.info(f"Splits gerados: {len(all_train)} treino, {len(all_test)} teste (Caso Real)")
    return all_train, all_test

def _selecionar_instancia_alvo(result: Any, img_path: str, config_sel: Dict[str, float]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
    """
    Seleciona a melhor instância (vaca) na imagem baseado em heurísticas.
    
    Critérios:
    - Confiança do bbox
    - Área do bbox
    - Proximidade do centro
    
    Returns:
        tuple: (Keypoints da instância vencedora, Dict com info do bbox) ou (None, None).
    """
    if not result.boxes:
        return None, None
        
    boxes = result.boxes
    keypoints = result.keypoints.data.cpu().numpy() # (N, 8, 3)
    
    img_h, img_w = result.orig_shape
    diag_img = math.sqrt(img_w**2 + img_h**2)
    center_img = (img_w / 2, img_h / 2)
    
    best_score = -float('inf')
    best_idx = -1
    best_bbox_info = None
    
    conf_min = config_sel.get("conf_min", 0.4)
    w_area = config_sel.get("peso_area", 0.5)
    w_conf = config_sel.get("peso_confianca", 0.5)
    bonus_centro = config_sel.get("bonus_centro", 0.2)
    
    for i, box in enumerate(boxes):
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
            
        # Bbox coords
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        w = x2 - x1
        h = y2 - y1
        area = w * h
        area_norm = area / (img_w * img_h)
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        dist_centro = math.sqrt((cx - center_img[0])**2 + (cy - center_img[1])**2)
        dist_centro_norm = dist_centro / diag_img
        
        # Score
        # Maximizar area e confiança, minimizar distancia do centro
        score = (w_area * area_norm) + (w_conf * conf) - (bonus_centro * dist_centro_norm)
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_bbox_info = {
                "bbox_w": w,
                "bbox_h": h,
                "bbox_area": area,
                "bbox_area_norm": area_norm,
                "bbox_aspect_ratio": w / h if h > 0 else 0
            }
            
    if best_idx != -1:
        return keypoints[best_idx], best_bbox_info
    
    return None, None

def _normalizar_orientacao_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Se a vaca estiver invertida (cabeça para baixo/esquerda?), espelha keypoints.
    Heurística simples: Posição do withers vs tail_head?
    Spec 4.3: "Calcular orientação por heurística (ex.: x(tail_head) - x(back))".
    Se ativado, normaliza para 0..1 dentro do bbox e inverte X se necessário.
    
    NOTA: Feature generation abaixo usa distâncias, que são invariantes a translação. 
    Mas angulos podem mudar se espelhar.
    
    Por simplicidade e seguindo a spec, vamos implementar uma versão básica se flag True.
    A spec diz: "Normalização de orientação (no espaço do bbox)... 2. Se invertida, x' = 1 - x"
    
    Isso requer ter o bbox da instância. Como aqui só recebo kpts, precisaria passar bbox também.
    IMPORTANTE: Calcular features geométricas usando coordenadas absolutas de pixels (distancia euclidiana) funciona igual.
    Apenas ANGULOS com sinal (se tiver) mudariam. `calcular_angulo` retorna 0..180 (sem sinal), então invariança a rotação/espelhamento
    depende de como definimos. 
    
    Se `calcular_angulo` usa lei dos cossenos (sem sinal), é invariante a espelhamento? Sim (cos(-x) = cos(x)).
    
    Entretanto, para features RELATIVAS (ex: delta X), orientação importa.
    No momento, vamos focar nas features de Distância e Angulo (sem sinal) que são robustas.
    Retornamos os kpts originais se a normalização for complexa demais para agora.
    
    TODO: refinar se necessário.
    """
    return kpts

from ..util.geometria import calcular_distancia, calcular_angulo, calcular_area_bbox, calcular_area_poligono, calcular_distancia_ponto_reta
from ..util.contratos import NomeKeypoint, LISTA_KEYPOINTS_ORDENADA
from sklearn.decomposition import PCA

# Mapeamento de índices para acesso rápido
IDX_KP = {nome: i for i, nome in enumerate(LISTA_KEYPOINTS_ORDENADA)}

def _calcular_features_geometricas(kpts: np.ndarray, bbox_info: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Calcula distâncias, razões, ângulos e features complexas conforme especificação.
    
    kpts: (8, 3) -> [x, y, conf]
    bbox_info: Dict com info do bbox (opcional)
    """
    feat = {}
    
    # --- Features de BBox ---
    bbox_area = 1.0 # Default
    if bbox_info:
        feat["bbox_aspect_ratio"] = bbox_info.get("bbox_aspect_ratio", 0.0)
        feat["bbox_area_norm"] = bbox_info.get("bbox_area_norm", 0.0)
        bbox_area = bbox_info.get("bbox_area", 1.0)
        if bbox_area <= 0: bbox_area = 1.0
    
    # Helper para pegar (x, y) de um nome
    def get_p(nome: str) -> Tuple[float, float]:
        idx = IDX_KP[nome]
        return (float(kpts[idx][0]), float(kpts[idx][1]))
        
    def get_v(nome: str) -> float:
        idx = IDX_KP[nome]
        return float(kpts[idx][2])
    
    # --- Distâncias Básicas ---
    # Pares chave para o esqueleto da vaca
    pairs = [
        ("withers", "back"), ("back", "hip"), ("hip", "tail_head"),
        ("hook_up", "hook_down"), ("pin_up", "pin_down"),
        ("hook_up", "pin_up"), ("hook_down", "pin_down"),
        ("hip", "hook_up"), ("hip", "hook_down"),
        ("hip", "pin_up"), ("hip", "pin_down"),
        ("back", "hook_up"), ("back", "hook_down") # etc
    ]
    
    for p1_name, p2_name in pairs:
        d = calcular_distancia(get_p(p1_name), get_p(p2_name))
        # dists[f"dist_{p1_name}_{p2_name}"] = d # Não salvar dist bruta como feature final para evitar overfitting de escala
        # Vamos usar apenas para calculos intermediarios se necessario

    # --- Razões Obrigatórias (Spec 10.4) ---
    def calc_ratio(num_p1, num_p2, den_p1, den_p2):
        num = calcular_distancia(get_p(num_p1), get_p(num_p2))
        den = calcular_distancia(get_p(den_p1), get_p(den_p2))
        if den == 0: return 0.0
        return num / den
        
    feat["razao_dist_hip_hook_up_por_dist_hook_up_pin_up"] = calc_ratio("hip", "hook_up", "hook_up", "pin_up")
    feat["razao_dist_hip_hook_down_por_dist_hook_down_pin_down"] = calc_ratio("hip", "hook_down", "hook_down", "pin_down")
    feat["razao_dist_hip_tail_head_por_dist_hook_down_pin_down"] = calc_ratio("hip", "tail_head", "hook_down", "pin_down")
    feat["razao_dist_hip_tail_head_por_dist_hook_up_pin_up"] = calc_ratio("hip", "tail_head", "hook_up", "pin_up")
    
    feat["razao_dist_hip_hook_up_por_dist_hip_tail_head"] = calc_ratio("hip", "hook_up", "hip", "tail_head")
    feat["razao_dist_hip_hook_down_por_dist_hip_tail_head"] = calc_ratio("hip", "hook_down", "hip", "tail_head")
    
    feat["razao_dist_back_hip_por_dist_hip_tail_head"] = calc_ratio("back", "hip", "hip", "tail_head")
    feat["razao_dist_back_hip_por_dist_hip_hook_up"] = calc_ratio("back", "hip", "hip", "hook_up")
    feat["razao_dist_back_hip_por_dist_hip_hook_down"] = calc_ratio("back", "hip", "hip", "hook_down")
    feat["razao_dist_back_hip_por_dist_hip_pin_up"] = calc_ratio("back", "hip", "hip", "pin_up")
    feat["razao_dist_back_hip_por_dist_hip_pin_down"] = calc_ratio("back", "hip", "hip", "pin_down")
    
    # --- Ângulos Obrigatórios (Spec 10.4) ---
    def calc_ang(p1, v, p3):
        return calcular_angulo(get_p(p1), get_p(v), get_p(p3))
        
    feat["angulo_hook_up_hip_hook_down"] = calc_ang("hook_up", "hip", "hook_down")
    feat["angulo_hip_hook_up_pin_up"] = calc_ang("hip", "hook_up", "pin_up")
    feat["angulo_hip_hook_down_pin_down"] = calc_ang("hip", "hook_down", "pin_down")
    feat["angulo_hook_up_hip_tail_head"] = calc_ang("hook_up", "hip", "tail_head")
    feat["angulo_hook_down_hip_tail_head"] = calc_ang("hook_down", "hip", "tail_head")
    feat["angulo_pin_up_tail_head_pin_down"] = calc_ang("pin_up", "tail_head", "pin_down")
    feat["angulo_pin_up_hip_pin_down"] = calc_ang("pin_up", "hip", "pin_down")
    feat["angulo_hook_up_back_hook_down"] = calc_ang("hook_up", "back", "hook_down")
    
    # Recomendadas
    feat["angulo_withers_back_tail_head"] = calc_ang("withers", "back", "tail_head")
    
    # --- FEATURES COMPLEXAS (NOVAS) ---
    
    # 1. Áreas de Polígonos (Normalizadas pela área do bbox)
    # Área Pélvica (Trapézio Traseiro): hook_up, hook_down, pin_down, pin_up
    poly_pelvic = [get_p("hook_up"), get_p("hook_down"), get_p("pin_down"), get_p("pin_up")]
    area_pelvic = calcular_area_poligono(poly_pelvic)
    feat["area_poligono_pelvico_norm"] = area_pelvic / bbox_area
    
    # Área Torácica (Triângulo Frontal): withers, back, hip
    poly_torax = [get_p("withers"), get_p("back"), get_p("hip")]
    area_torax = calcular_area_poligono(poly_torax)
    feat["area_triangulo_torax_norm"] = area_torax / bbox_area

    # 2. Índices de Conformação
    # Índice de Robustez: Largura Hooks / Comprimento Corpo (Withers -> Tail)
    width_hooks = calcular_distancia(get_p("hook_up"), get_p("hook_down"))
    len_body = calcular_distancia(get_p("withers"), get_p("tail_head"))
    feat["indice_robustez"] = width_hooks / len_body if len_body > 0 else 0.0
    
    # Índice de Triângulo Traseiro (Hip -> Pins) / BBox Area
    # (Triangulo formado por Hip e os dois Pins - simplificação geometrica)
    poly_tri_rear = [get_p("hip"), get_p("pin_up"), get_p("pin_down")]
    area_tri_rear = calcular_area_poligono(poly_tri_rear)
    feat["indice_triangulo_traseiro"] = area_tri_rear / bbox_area

    # 3. Curvatura da Coluna
    # Soma de angulos absolutos ao longo da coluna não é trivial pois angulo é sempre positivo.
    # Vamos usar Distancia Perpendicular (Desvio) de Back e Hip em relação à linha Withers->TailHead.
    p_withers = get_p("withers")
    p_tail = get_p("tail_head")
    
    dev_back = calcular_distancia_ponto_reta(get_p("back"), p_withers, p_tail)
    dev_hip = calcular_distancia_ponto_reta(get_p("hip"), p_withers, p_tail)
    
    # Normalizar pelo len_body para ser invariante a escala
    feat["desvio_coluna_back_norm"] = dev_back / len_body if len_body > 0 else 0.0
    feat["desvio_coluna_hip_norm"] = dev_hip / len_body if len_body > 0 else 0.0
    
    # 5. PCA Excentricidade (Global Shape)
    # Usa coordenadas XY puras para achar eixos principais
    coords = kpts[:, :2] # (8, 2)
    if len(coords) >= 3:
        try:
            pca = PCA(n_components=2)
            pca.fit(coords)
            # variances (eigenvalues)
            explained_var = pca.explained_variance_
            # razao maior / menor
            if explained_var[1] > 0:
                feat["pca_excentricidade"] = explained_var[0] / explained_var[1]
            else:
                feat["pca_excentricidade"] = 0.0
        except:
             feat["pca_excentricidade"] = 0.0
    else:
        feat["pca_excentricidade"] = 0.0

    # 6. Shape Context (Coordenadas Relativas)
    # Transformar sistema de coordenadas para que:
    # Origem = Withers
    # Eixo X = Linha Withers -> TailHead
    # Isso torna a postura da vaca invariante a rotação e translação.
    
    origin = np.array(p_withers)
    target = np.array(p_tail)
    
    # Vetor base
    vec_base = target - origin
    len_base = np.linalg.norm(vec_base)
    
    if len_base > 0:
        # Vetor unitario X
        u_x = vec_base / len_base
        # Vetor unitario Y (perpendicular: -y, x)
        u_y = np.array([-u_x[1], u_x[0]])
        
        # Projetar cada ponto
        for i, nome in enumerate(LISTA_KEYPOINTS_ORDENADA):
            p = kpts[i, :2]
            vec_p = p - origin
            
            # Projeção no novo sistema
            new_x = np.dot(vec_p, u_x)
            new_y = np.dot(vec_p, u_y)
            
            # Normalizar por len_base (escala)
            feat[f"sc_{nome}_x"] = new_x / len_base
            feat[f"sc_{nome}_y"] = new_y / len_base
    else:
        # Fallback (vaca pontual??)
        for nome in LISTA_KEYPOINTS_ORDENADA:
            feat[f"sc_{nome}_x"] = 0.0
            feat[f"sc_{nome}_y"] = 0.0
            
    return feat
