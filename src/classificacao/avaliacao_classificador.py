import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

from ..util.io_arquivos import garantir_diretorio


def _ler_cfg_rejeicao_predicao(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Le e normaliza as configuracoes de rejeicao de predicao.

    Args:
        config (Dict[str, Any]): Configuracao global.

    Returns:
        Dict[str, Any]: Configuracao efetiva de rejeicao.
    """
    cfg = config.get("classificacao", {}).get("rejeicao_predicao", {})
    return {
        "habilitar": bool(cfg.get("habilitar", False)),
        "confianca_min": float(cfg.get("confianca_min", 0.35)),
        "margem_top1_top2_min": float(cfg.get("margem_top1_top2_min", 0.0)),
        "rotulo_nao_identificado": str(cfg.get("rotulo_nao_identificado", "NAO_IDENTIFICADO")),
    }


def avaliar_classificador(config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    avaliar_classificador: Avalia o modelo no conjunto de teste (10%).
    Oscilação da matriz de confusão e métricas finais.

    Parametros:
        config (Dict[str, Any]): Configuracoes gerais do projeto.
        logger (logging.Logger): Logger para mensagens de execucao.

    Retorno:
        None: Salva relatorios em disco e imprime metricas no log.
    """
    logger.info("=== Fase 3: Avaliação do Classificador (Teste Final) ===")
    
    processed_dir = Path(config["paths"]["processed"]) / "classificacao"
    features_csv = processed_dir / "features" / "features_completas.csv"
    splits_dir = processed_dir / "splits"
    test_txt = splits_dir / "teste_10pct.txt"
    
    models_dir = Path(config["paths"]["models"]) / "classificacao" / "modelos_salvos"
    reports_dir = Path(config["paths"]["outputs"]) / "relatorios"
    garantir_diretorio(reports_dir)
    model_type = config.get("classificacao", {}).get("modelo_padrao", "xgboost")
    
    # Validar arquivos
    if not features_csv.exists() or not test_txt.exists():
        logger.error("Arquivos de dados não encontrados. Rode as etapas anteriores.")
        return

    if model_type == "catboost":
        model_path = models_dir / "catboost_model.cbm"
    elif model_type == "sklearn_rf":
        model_path = models_dir / "rf_model.joblib"
    elif model_type == "svm":
        model_path = models_dir / "svm_model.joblib"
    elif model_type == "mlp":
        model_path = models_dir / "mlp_model.joblib"
    else:
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
    if model_type == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier()
        clf.load_model(str(model_path))
    elif model_type in ("sklearn_rf", "svm", "mlp"):
        clf = joblib.load(model_path)
    else:
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
    
    # Inferência
    preds = np.asarray(clf.predict(X_test)).reshape(-1)
    probs = np.asarray(clf.predict_proba(X_test))
    if probs.ndim == 1:
        probs = np.vstack([1.0 - probs, probs]).T
    conf_max = probs.max(axis=1).reshape(-1)
    probs_sorted = np.sort(probs, axis=1)
    conf_top2 = probs_sorted[:, -2] if probs.shape[1] > 1 else np.zeros_like(conf_max)
    margem_top1_top2 = conf_max - conf_top2
    acertos = (preds == y_test).reshape(-1)
    
    # Métricas
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro")
    topk_metrics = {}
    for k in (1, 3, 5):
        k_eff = min(k, probs.shape[1])
        topk_idx = np.argsort(probs, axis=1)[:, ::-1][:, :k_eff]
        topk_ok = np.any(topk_idx == y_test.reshape(-1, 1), axis=1)
        topk_metrics[f"top{k_eff}_accuracy"] = float(np.mean(topk_ok))
    logger.info(
        f"RESULTADO FINAL (Teste): Acurácia: {acc:.4f}, Precision-Macro: {prec:.4f}, "
        f"Recall-Macro: {rec:.4f}, F1-Macro: {f1:.4f}"
    )

    # Top-k independe da regra de rejeicao.
    top1 = float(topk_metrics.get("top1_accuracy", topk_metrics.get("top1", acc)))
    top3 = float(topk_metrics.get("top3_accuracy", topk_metrics.get("top3", 0.0)))
    top5 = float(topk_metrics.get("top5_accuracy", topk_metrics.get("top5", 0.0)))
    logger.info("top1: %.4f", top1)
    logger.info("top3: %.4f", top3)
    logger.info("top5: %.4f", top5)

    # Metricas com rejeicao opcional (NAO_IDENTIFICADO)
    cfg_rejeicao = _ler_cfg_rejeicao_predicao(config)
    resumo_rejeicao = None
    if cfg_rejeicao["habilitar"]:
        aceitas = (conf_max >= cfg_rejeicao["confianca_min"]) & (
            margem_top1_top2 >= cfg_rejeicao["margem_top1_top2_min"]
        )
        cobertura_rejeicao = float(np.mean(aceitas))
        rejeitadas = int(np.sum(~aceitas))
        aceitas_n = int(np.sum(aceitas))
        if aceitas_n > 0:
            acc_aceitas = float(np.mean(acertos[aceitas]))
            prec_aceitas = float(precision_score(y_test[aceitas], preds[aceitas], average="macro", zero_division=0))
            rec_aceitas = float(recall_score(y_test[aceitas], preds[aceitas], average="macro", zero_division=0))
            f1_aceitas = float(f1_score(y_test[aceitas], preds[aceitas], average="macro", zero_division=0))
        else:
            acc_aceitas = 0.0
            prec_aceitas = 0.0
            rec_aceitas = 0.0
            f1_aceitas = 0.0
        resumo_rejeicao = {
            "confianca_min": cfg_rejeicao["confianca_min"],
            "margem_top1_top2_min": cfg_rejeicao["margem_top1_top2_min"],
            "cobertura": cobertura_rejeicao,
            "rejeitadas": rejeitadas,
            "aceitas": aceitas_n,
            "metricas_aceitas": {
                "accuracy": acc_aceitas,
                "precision_macro": prec_aceitas,
                "recall_macro": rec_aceitas,
                "f1_macro": f1_aceitas,
            },
        }
        logger.info(
            "COM REJEICAO - Cobertura: %.4f | Rejeitadas: %d | Acuracia(aceitas): %.4f | F1(aceitas): %.4f",
            cobertura_rejeicao,
            rejeitadas,
            acc_aceitas,
            f1_aceitas,
        )

    if cfg_rejeicao["habilitar"] and resumo_rejeicao is not None:
        logger.info("Com rejeicao (confianca_min=%.2f)", cfg_rejeicao["confianca_min"])
        logger.info(
            "cobertura (percentual de imagens aceitas pelos limiares de confianca top1 e margem top1-top2): %.4f",
            float(resumo_rejeicao.get("cobertura", 0.0)),
        )
        metricas_aceitas = resumo_rejeicao.get("metricas_aceitas", {})
        logger.info("accuracy nas aceitas: %.4f", float(metricas_aceitas.get("accuracy", 0.0)))
        logger.info("f1_macro nas aceitas: %.4f", float(metricas_aceitas.get("f1_macro", 0.0)))
    
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
        "rejeicao_predicao": resumo_rejeicao,
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
        """Encoder JSON para serializar tipos NumPy em tipos nativos Python."""

        def default(self, obj: Any) -> Any:
            """
            Converte objetos NumPy para representacoes serializaveis em JSON.

            Parametros:
                obj (Any): Objeto a ser serializado.

            Retorno:
                Any: Valor convertido para tipo serializavel em JSON.
            """
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(reports_dir / "metricas_classificacao.json", 'w') as f:
        json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)
