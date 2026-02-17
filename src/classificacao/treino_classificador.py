import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

from ..util.io_arquivos import garantir_diretorio

def treinar_classificador(config: Dict[str, Any], logger: logging.Logger) -> Path:
    """
    treinar_classificador: Treina o modelo de classificação (Fase 3).
    
    Carrega features_completas.csv e splits/treino.txt.
    Treina XGBoost (ou RF) com validação interna e early stopping.
    Salva modelo e métricas.

    Args:
        config (Dict[str, Any]): Configurações.
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho do modelo salvo.
    """
    logger.info("=== Fase 3: Treino Classificador ===")
    
    # Caminhos
    processed_dir = Path(config["paths"]["processed"]) / "classificacao"
    features_csv = processed_dir / "features" / "features_completas.csv"
    splits_dir = processed_dir / "splits"
    train_txt = splits_dir / "treino.txt"
    
    models_dir = Path(config["paths"]["models"]) / "classificacao" / "modelos_salvos"
    garantir_diretorio(models_dir)
    
    reports_dir = Path(config["paths"]["outputs"]) / "relatorios"
    garantir_diretorio(reports_dir)
    
    # Validar existência
    if not features_csv.exists():
        logger.error(f"Arquivo de features não encontrado: {features_csv}")
        raise FileNotFoundError(f"Arquivo de features não encontrado: {features_csv}")
        
    if not train_txt.exists():
        logger.error(f"Arquivo de split treino.txt não encontrado: {train_txt}")
        raise FileNotFoundError(f"Split treino.txt não encontrado. Rode 'gerar-features'.")
        
    # Carregar Dados
    df = pd.read_csv(features_csv)
    with open(train_txt, 'r') as f:
        train_files = set(line.strip() for line in f if line.strip())
        
    # Filtrar apenas dados de TREINO (Spec 12: Data leakage policy)
    # O df tem coluna 'arquivo'
    df_train = df[df['arquivo'].isin(train_files)].copy()
    
    if df_train.empty:
        logger.error("Dataset de treino vazio após filtro pelo split.")
        raise ValueError("Dataset de treino vazio.")
        
    logger.info(f"Dataset carregado. Total features: {len(df)}. Treino filtrado: {len(df_train)}.")
    
    # Preparar X e y
    target_col = "classe"
    exclude_cols = ["arquivo", "classe", "target"] # target as vezes existe
    
    # Seleção de features pelo config
    feats_cfg = config.get("classificacao", {}).get("features", {}).get("selecionadas", "todas")
    
    if isinstance(feats_cfg, list):
        feature_cols = feats_cfg
        # Validar se colunas existem
        missing = [c for c in feature_cols if c not in df_train.columns]
        if missing:
            logger.warning(f"Features configuradas não encontradas no CSV: {missing}")
            feature_cols = [c for c in feature_cols if c in df_train.columns]
    else:
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]
        
    logger.info(f"Usando {len(feature_cols)} features: {feature_cols}")
    
    X = df_train[feature_cols].copy()
    y_raw = df_train[target_col].copy()
    
    # Tratar NaN e Infinitos
    # Algumas razões podem ser NaN/Inf se denominador for zero ou keypoint sumir.
    # XGBoost lida com NaN nativamente, mas sklearn RF não.
    # Vamos preencher com -1 ou média?
    # Para consistência, vamos preencher NaN com 0 (indicando 'ausência' de razão válida ou feature zerada)
    X = X.fillna(0.0)
    X = X.replace([np.inf, -np.inf], 0.0)
    
    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes_name = le.classes_
    logger.info(f"Classes identificadas: {len(classes_name)}")
    
    # Salvar LabelEncoder para inferência
    joblib.dump(le, models_dir / "label_encoder.pkl")
    
    # Configurar Modelo
    cls_config = config.get("classificacao", {})
    model_type = cls_config.get("modelo_padrao", "xgboost")
    
    if model_type == "xgboost":
        return _treinar_xgboost(X, y, cls_config, models_dir, reports_dir, feature_cols, logger)
    elif model_type == "sklearn_rf":
        return _treinar_rf(X, y, cls_config, models_dir, reports_dir, logger)
    else:
        logger.warning(f"Modelo {model_type} desconhecido. Usando XGBoost.")
        return _treinar_xgboost(X, y, cls_config, models_dir, reports_dir, feature_cols, logger)

def _treinar_xgboost(X, y, config_cls, models_dir, reports_dir, feature_cols, logger):
    # Split interno para validação/early stopping (Spec 11.1)
    val_frac = config_cls.get("validacao_interna", {}).get("fracao", 0.2)
    stratify = config_cls.get("validacao_interna", {}).get("estratificar", True)
    stratify_y = y if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_frac, random_state=42, stratify=stratify_y)
    
    logger.info(f"Treino interno: {len(X_train)} | Validação interna: {len(X_val)}")
    
    early_stopping_rounds = config_cls.get("validacao_interna", {}).get("early_stopping_rounds", 50)
    
    # Modelo
    # Ajustar para multi-classe
    num_class = len(np.unique(y))
    
    clf = xgb.XGBClassifier(
        n_estimators=1000, # Alto para early stopping cortar
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_class,
        random_state=42,
        tree_method="hist", # Mais rapido
        device="cuda" if config_cls.get("device", "cpu") != "cpu" else "cpu",
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds
    )
    
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    logger.info("Iniciando treino XGBoost com Early Stopping...")
    
    # ATENÇÃO: fit params de early_stopping mudaram nas versoes novas do XGBoost (>=2.0? scikit-learn wrapper)
    # Tentar padrao atual
    try:
        clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
    except TypeError:
        # Fallback para versoes antigas ou novas args
        clf.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
    best_iter = clf.best_iteration
    logger.info(f"Melhor iteração: {best_iter}")
    
    # Avaliar no Val interno (só pra log)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    logger.info(f"Validação Interna - Acurácia: {acc:.4f}, F1-Macro: {f1:.4f}")
    
    # Salvar
    model_path = models_dir / "xgboost_model.json"
    clf.save_model(model_path)
    logger.info(f"Modelo salvo em {model_path}")
    
    # Salvar features names (XGBoost JSON salva, mas bom ter separado se mudar pra pickle)
    joblib.dump(feature_cols, models_dir / "feature_names.pkl")
    
    # Relatório Json
    metrics = {
        "modelo": "xgboost",
        "best_iteration": best_iter,
        "internal_validation": {
            "accuracy": acc,
            "f1_macro": f1
        },
        "classification_report": classification_report(y_val, preds, output_dict=True, zero_division=0)
    }
    
    with open(reports_dir / "metricas_classificacao_treino.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    return model_path

def _treinar_rf(X, y, config_cls, models_dir, reports_dir, logger):
    logger.info("Treinando RandomForest (Baseline)...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    
    # Sem early stopping nativo da mesma forma, usa CV ou OOB
    val_frac = config_cls.get("validacao_interna", {}).get("fracao", 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_frac, random_state=42, stratify=y)
    
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    logger.info(f"RF Validação Interna - Acurácia: {acc:.4f}")
    
    model_path = models_dir / "rf_model.joblib"
    joblib.dump(clf, model_path)
    
    return model_path
