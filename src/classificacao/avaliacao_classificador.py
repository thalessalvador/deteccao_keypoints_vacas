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
    if "origem_instancia" in df_test.columns:
        df_test = df_test[df_test["origem_instancia"] == "real"].copy()
    
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
        exclude_cols = [
            "arquivo", "classe", "target",
            "origem_instancia", "is_aug", "aug_id", "split_instancia"
        ]
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
    conf_max = probs.max(axis=1)
    acertos = preds == y_test
    
    # Métricas
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    topk_metrics = {}
    for k in (1, 3, 5):
        k_eff = min(k, probs.shape[1])
        topk_idx = np.argsort(probs, axis=1)[:, ::-1][:, :k_eff]
        topk_ok = np.any(topk_idx == y_test.reshape(-1, 1), axis=1)
        topk_metrics[f"top{k_eff}_accuracy"] = float(np.mean(topk_ok))
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

    # Metricas por classe (precision/recall/f1)
    report = classification_report(y_test, preds, target_names=classes, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T
    df_classes = df_report.loc[classes, ["precision", "recall", "f1-score"]]
    ax = df_classes.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Metricas por Classe (Precision/Recall/F1)")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(reports_dir / "metricas_por_classe.png")
    plt.close()

    # Confianca: corretas vs incorretas
    plt.figure(figsize=(10, 6))
    if np.any(acertos):
        sns.histplot(conf_max[acertos], color="green", label="Corretas", bins=20, stat="density", alpha=0.45)
    if np.any(~acertos):
        sns.histplot(conf_max[~acertos], color="red", label="Incorretas", bins=20, stat="density", alpha=0.45)
    plt.title("Distribuicao da Confianca (Corretas vs Incorretas)")
    plt.xlabel("Confianca maxima da predicao")
    plt.ylabel("Densidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "confianca_corretas_vs_incorretas.png")
    plt.close()

    # Cobertura x acuracia por threshold de confianca
    thresholds = np.linspace(0.0, 1.0, 21)
    cobertura = []
    acuracia_filtrada = []
    for t in thresholds:
        mask = conf_max >= t
        cov = float(np.mean(mask))
        cobertura.append(cov)
        if np.any(mask):
            acc_t = float(np.mean(acertos[mask]))
        else:
            acc_t = np.nan
        acuracia_filtrada.append(acc_t)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, cobertura, label="Cobertura")
    plt.plot(thresholds, acuracia_filtrada, label="Acuracia (amostras aceitas)")
    plt.title("Cobertura vs Acuracia por Threshold de Confianca")
    plt.xlabel("Threshold de confianca")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "cobertura_vs_acuracia.png")
    plt.close()
    
    logger.info(f"Matriz de confusão salva em {reports_dir}")
    
    # Relatório JSON Final
    report = classification_report(y_test, preds, target_names=classes, output_dict=True, zero_division=0)
    final_metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "top_k_accuracy": topk_metrics,
        "confidence_analysis": {
            "media_conf_corretas": float(np.mean(conf_max[acertos])) if np.any(acertos) else None,
            "media_conf_incorretas": float(np.mean(conf_max[~acertos])) if np.any(~acertos) else None
        },
        "coverage_vs_accuracy": [
            {
                "threshold": float(t),
                "coverage": float(c),
                "accuracy_accepted": (None if np.isnan(a) else float(a))
            }
            for t, c, a in zip(thresholds, cobertura, acuracia_filtrada)
        ],
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
