import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
import xgboost as xgb

from ..util.io_arquivos import garantir_diretorio
from ..classificacao.gerador_dataset_features import _selecionar_instancia_alvo, _normalizar_orientacao_keypoints, _calcular_features_geometricas

def classificar_imagem_unica(config: Dict[str, Any], img_path: Path, top_k: int = 3, desenhar: bool = False, logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Realiza a classificação de uma única imagem end-to-end:
    1. Detecta pose (YOLO)
    2. Extrai features geométricas
    3. Classifica (XGBoost)
    
    Returns:
        Dict com 'classe_predita', 'confianca', 'top_k', 'features', 'keypoints'
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Iniciando classificação para: {img_path}")
    
    # 1. Carregar Modelo de Pose
    # (Reciclando lógica de achar melhor modelo ou config)
    pose_cfg = config.get("pose", {})
    runs_dir = Path("modelos/pose/runs").resolve()
    
    model_path = None
    # Tentar pegar do relatorio
    try:
        relatorio_path = Path("saidas/relatorios/metricas_pose.json").resolve()
        if relatorio_path.exists():
            with open(relatorio_path, 'r') as f:
                data = json.load(f)
                model_path = data.get("melhor_modelo", {}).get("path")
    except Exception as e:
        logger.warning(f"Nao foi possivel ler metricas_pose.json: {e}")
        
    if not model_path or not Path(model_path).exists():
        candidates = list(runs_dir.rglob("best.pt"))
        if candidates:
            model_path = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
        else:
            model_path = pose_cfg.get("model_name", "yolov8n-pose.pt")
            
    logger.info(f"Carregando modelo de pose: {model_path}")
    pose_model = YOLO(model_path)
    
    # 2. Carregar Classificador e Artefatos
    cls_model_dir = Path("modelos/classificacao/modelos_salvos")
    model_type = config.get("classificacao", {}).get("modelo_padrao", "xgboost")
    if model_type == "catboost":
        model_path_cls = cls_model_dir / "catboost_model.cbm"
    elif model_type == "sklearn_rf":
        model_path_cls = cls_model_dir / "rf_model.joblib"
    else:
        model_path_cls = cls_model_dir / "xgboost_model.json"
    le_path = cls_model_dir / "label_encoder.pkl"
    feats_path = cls_model_dir / "feature_names.pkl"
    
    if not model_path_cls.exists() or not le_path.exists() or not feats_path.exists():
        msg = "Modelos de classificação não encontrados. Execute 'treinar-classificador' primeiro."
        logger.error(msg)
        return {"erro": msg}
        
    # Carregar modelo de classificacao
    if model_type == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier()
        clf.load_model(str(model_path_cls))
    elif model_type == "sklearn_rf":
        import joblib
        clf = joblib.load(model_path_cls)
    else:
        clf = xgb.XGBClassifier()
        clf.load_model(str(model_path_cls))
    
    # Carregar LabelEncoder e Feature Names
    import joblib
    le = joblib.load(le_path)
    feature_names = joblib.load(feats_path) # Lista de nomes de colunas esperados
        
    # 3. Inferência de Pose
    results = pose_model.predict(source=str(img_path), save=False, conf=0.25, verbose=False)
    result = results[0]
    
    # Selecionar melhor vaca
    sel_cfg = config.get("classificacao", {}).get("selecao_instancia", {})
    instancia_kpts, instancia_bbox = _selecionar_instancia_alvo(result, str(img_path), sel_cfg)
    
    if instancia_kpts is None:
        msg = "Nenhuma vaca detectada com confiança suficiente."
        logger.warning(msg)
        return {"erro": msg, "detecao_pose": False}
        
    # 4. Calcular Features
    kpts = instancia_kpts
    if config.get("classificacao", {}).get("normalizar_orientacao", False):
        kpts = _normalizar_orientacao_keypoints(kpts)
        
    feats_dict = _calcular_features_geometricas(kpts, instancia_bbox)
    
    # Criar DataFrame com APENAS as features esperadas pelo modelo
    # Preencher com 0.0 se alguma feature nova não estiver no dicionario (embora deva estar)
    df_input = pd.DataFrame([feats_dict])
    
    # Garantir ordem e colunas corretas
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0.0 # ou np.nan
            
    df_input = df_input[feature_names] # Reordenar
    
    # 5. Predição
    probs = clf.predict_proba(df_input)[0] # Array de probabilidades
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confianca = float(probs[pred_idx])
    
    # Top K
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_preds = []
    for idx in top_indices:
        label = le.inverse_transform([idx])[0]
        prob = float(probs[idx])
        top_preds.append({"classe": label, "confianca": prob})
        
    logger.info(f"Classificação: {pred_label} ({confianca:.2%})")
    
    retorno = {
        "arquivo": img_path.name,
        "classe_predita": pred_label,
        "confianca": confianca,
        "top_k": top_preds,
        "features": feats_dict,
        "bbox": instancia_bbox
    }
    
    # 6. Desenhar (Opcional)
    if desenhar:
        # Carregar imagem original
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Falha ao ler imagem para desenho.")
        else:
            # Desenhar Bbox
            if instancia_bbox:
                # Recuperar coords do bbox original? O _selecionar retorna bbox processado (w,h).
                # Precisariamos do xyxy original. O _selecionar_instancia_alvo atual retorna Dict com w, h, area.
                # Para desenhar precisamos do box original.
                # Vamos re-iterar sobre results.boxes para achar o que deu match? 
                # Ou simplificar e desenhar o plot do YOLO methods.
                
                # Melhor: Usar o plot() do ultralytics na instancia?
                # Como selecionamos UMA, podemos filtrar.
                pass

            # Desenhar Keypoints e Skeleton
            # Iterar kpts e desenhar
            # (Simplificação: desenhar label no canto)
            cv2.putText(img, f"Pred: {pred_label} ({confianca:.1%})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Listar Top 3
            y = 70
            for item in top_preds[1:]:
                cv2.putText(img, f"{item['classe']}: {item['confianca']:.1%}", (10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            out_dir = Path(config["paths"]["outputs"]) / "inferencias" / "classificacao"
            garantir_diretorio(out_dir)
            out_path = out_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_path), img)
            logger.info(f"Imagem classificada salva em: {out_path}")
            retorno["imagem_salva"] = str(out_path)

    return retorno
