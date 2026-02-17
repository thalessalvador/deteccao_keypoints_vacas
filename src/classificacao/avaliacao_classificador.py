import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

from ..util.io_arquivos import garantir_diretorio

def avaliar_classificador(config: Dict[str, Any], logger: logging.Logger):
    """
    avaliar_classificador: Avalia o modelo no conjunto de teste (10%).
    Oscilação da matriz de confusão e métricas finais.
    """
    logger.info("=== Fase 3: Avaliação do Classificador (Teste Final) ===")
    
    processed_dir = Path(config["paths"]["processed"]) / "classificacao"
    features_csv = processed_dir / "features" / "features_completas.csv"
    splits_dir = processed_dir / "splits"
    test_txt = splits_dir / "teste_10pct.txt"
    
    models_dir = Path(config["paths"]["models"]) / "classificacao" / "modelos_salvos"
    reports_dir = Path(config["paths"]["outputs"]) / "relatorios"
    garantir_diretorio(reports_dir)
    
    # Validar arquivos
    if not features_csv.exists() or not test_txt.exists():
        logger.error("Arquivos de dados não encontrados. Rode as etapas anteriores.")
        return

    model_path = models_dir / "xgboost_model.json"
    le_path = models_dir / "label_encoder.pkl"
    fn_path = models_dir / "feature_names.pkl" # Se existir
    
    if not model_path.exists() or not le_path.exists():
        logger.error(f"Modelo não encontrado em {models_dir}. Treine antes.")
        return

    # Carregar Dados
    df = pd.read_csv(features_csv)
    with open(test_txt, 'r') as f:
        test_files = set(line.strip() for line in f if line.strip())
        
    df_test = df[df['arquivo'].isin(test_files)].copy()
    
    if df_test.empty:
        logger.error("Dataset de teste vazio.")
        return
        
    logger.info(f"Dataset de teste: {len(df_test)} instâncias.")
    
    # Carregar LabelEncoder
    le = joblib.load(le_path)
    
    # Preparar X e y
    # Precisamos usar as MESMAS features do treino
    if fn_path.exists():
        feature_cols = joblib.load(fn_path)
    else:
        # Tentar inferir ou usar config (arriscado se mudou)
        # Vamos assumir que config é a fonte da verdade se pkl não existe
        exclude_cols = ["arquivo", "classe", "target"]
        feats_cfg = config.get("classificacao", {}).get("features", {}).get("selecionadas", "todas")
        if isinstance(feats_cfg, list):
            feature_cols = feats_cfg
            feature_cols = [c for c in feature_cols if c in df_test.columns] # Validar
        else:
            feature_cols = [c for c in df_test.columns if c not in exclude_cols]
            
    logger.info(f"Features usadas: {len(feature_cols)}")
    
    X_test = df_test[feature_cols].copy()
    X_test = X_test.fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], 0.0)
    
    y_test_str = df_test["classe"].values
    y_test = le.transform(y_test_str)
    
    # Carregar Modelo
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    
    # Inferência
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    logger.info(f"RESULTADO FINAL (Teste): Acurácia: {acc:.4f}, F1-Macro: {f1:.4f}")
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, preds)
    classes = [str(c) for c in le.classes_] # Garantir string para JSON keys
    
    # Salvar CSV da Matriz
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    df_cm.to_csv(reports_dir / "matriz_confusao.csv")
    
    # Plotar
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão - Caso Real (10%)")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()
    plt.savefig(reports_dir / "matriz_confusao.png")
    plt.close()
    
    logger.info(f"Matriz de confusão salva em {reports_dir}")
    
    # Relatório JSON Final
    report = classification_report(y_test, preds, target_names=classes, output_dict=True, zero_division=0)
    final_metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "classification_report": report
    }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(reports_dir / "metricas_classificacao.json", 'w') as f:
        json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)
