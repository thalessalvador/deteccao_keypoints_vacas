import logging
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit

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
    exclude_cols = [
        "arquivo", "classe", "target",
        "origem_instancia", "is_aug", "aug_id", "split_instancia"
    ]
    
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
    origem_instancia = (
        df_train["origem_instancia"].astype(str).values
        if "origem_instancia" in df_train.columns
        else np.array(["real"] * len(df_train), dtype=object)
    )
    groups = df_train["arquivo"].astype(str).values
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
    
    aug_stats = _resumir_augmentation_em_df(df_train)
    logger.info(
        f"Instancias classificador (split treino) - reais: {aug_stats['instancias_reais']} | "
        f"augmentadas: {aug_stats['instancias_augmentadas']} | total: {aug_stats['instancias_total']}"
    )

    # Configurar Modelo
    cls_config = config.get("classificacao", {})
    model_type = cls_config.get("modelo_padrao", "xgboost")
    
    if model_type == "xgboost":
        return _treinar_xgboost(X, y, groups, origem_instancia, cls_config, models_dir, reports_dir, feature_cols, aug_stats, logger)
    elif model_type == "sklearn_rf":
        return _treinar_rf(X, y, cls_config, models_dir, reports_dir, logger)
    else:
        logger.warning(f"Modelo {model_type} desconhecido. Usando XGBoost.")
        return _treinar_xgboost(X, y, groups, origem_instancia, cls_config, models_dir, reports_dir, feature_cols, aug_stats, logger)

def _treinar_xgboost(X, y, groups, origem_instancia, config_cls, models_dir, reports_dir, feature_cols, aug_stats, logger):
    # Split interno para validacao/early stopping (Spec 11.1)
    # Importante: split por grupo (arquivo) evita leakage entre copias augmentadas da mesma imagem.
    val_frac = config_cls.get("validacao_interna", {}).get("fracao", 0.2)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    usar_apenas_real_val = bool(config_cls.get("validacao_interna", {}).get("usar_apenas_real", False))
    if usar_apenas_real_val:
        val_mask_real = origem_instancia[val_idx] == "real"
        val_idx_filtrado = val_idx[val_mask_real]
        if len(val_idx_filtrado) > 0:
            val_idx = val_idx_filtrado
            logger.info(f"Validacao interna configurada para usar apenas instancias reais: {len(val_idx)} amostras.")
        else:
            logger.warning("validacao_interna.usar_apenas_real=true, mas nenhuma instancia real caiu na validacao. Usando validacao completa.")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    groups_train = set(groups[train_idx].tolist())
    groups_val = set(groups[val_idx].tolist())
    overlap_groups = len(groups_train.intersection(groups_val))
    
    logger.info(f"Treino interno: {len(X_train)} | Validação interna: {len(X_val)}")
    logger.info(
        f"Split interno por grupo(arquivo): grupos_treino={len(groups_train)} | "
        f"grupos_validacao={len(groups_val)} | sobreposicao={overlap_groups}"
    )
    
    logger.info(
        f"Instancias classificador (usadas para treino) - total: {len(X_train)} | validacao interna: {len(X_val)}"
    )
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
        eval_metric=["mlogloss", "merror"],
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
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec = recall_score(y_val, preds, average="macro", zero_division=0)
    f1 = f1_score(y_val, preds, average="macro")
    logger.info(
        f"Validação Interna - Acurácia: {acc:.4f}, Precision-Macro: {prec:.4f}, "
        f"Recall-Macro: {rec:.4f}, F1-Macro: {f1:.4f}"
    )
    
    # Salvar
    model_path = models_dir / "xgboost_model.json"
    clf.save_model(model_path)
    logger.info(f"Modelo salvo em {model_path}")
    
    # Salvar features names (XGBoost JSON salva, mas bom ter separado se mudar pra pickle)
    joblib.dump(feature_cols, models_dir / "feature_names.pkl")

    # ImportÃ¢ncia das features para anÃ¡lise do que mais contribuiu no modelo
    top_n_imp = config_cls.get("top_n_importancias", 20)
    feature_importance = _extrair_importancia_xgboost(clf, feature_cols, top_n=top_n_imp)
    _salvar_graficos_xgboost(clf, feature_importance, reports_dir, logger)
    
    # Relatório Json
    metrics = {
        "modelo": "xgboost",
        "best_iteration": best_iter,
        "internal_validation": {
            "accuracy": acc,
            "f1_macro": f1
        },
        "augmentation_stats": aug_stats,
        "feature_importance": feature_importance,
        "classification_report": classification_report(y_val, preds, output_dict=True, zero_division=0)
    }
    
    with open(reports_dir / "metricas_classificacao_treino.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    return model_path


def _resumir_augmentation_em_df(df_train: pd.DataFrame) -> Dict[str, Any]:
    if "origem_instancia" not in df_train.columns:
        total = int(len(df_train))
        return {
            "baseado_em_coluna_origem": False,
            "instancias_reais": total,
            "instancias_augmentadas": 0,
            "instancias_total": total,
        }

    reais = int((df_train["origem_instancia"] == "real").sum())
    aug = int((df_train["origem_instancia"] == "augmentation").sum())
    return {
        "baseado_em_coluna_origem": True,
        "instancias_reais": reais,
        "instancias_augmentadas": aug,
        "instancias_total": int(len(df_train)),
    }


def _extrair_importancia_xgboost(clf: xgb.XGBClassifier, feature_cols: List[str], top_n: int = 20) -> Dict[str, Any]:
    """
    Extrai importÃ¢ncia de features do XGBoost usando diferentes critÃ©rios.

    Retorna rankings por:
    - gain: ganho mÃ©dio das divisÃµes usando a feature
    - weight: quantidade de vezes que a feature foi usada
    - cover: cobertura mÃ©dia das divisÃµes
    - sklearn: feature_importances_ do wrapper sklearn
    """
    booster = clf.get_booster()
    fmap = {f"f{i}": nome for i, nome in enumerate(feature_cols)}
    feature_set = set(feature_cols)
    col_to_idx = {nome: i for i, nome in enumerate(feature_cols)}

    def normalizar_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normaliza chaves vindas do booster.get_score().

        O XGBoost pode retornar:
        - 'f0', 'f1', ... (quando usa índice interno)
        - nome da coluna (quando treinado com DataFrame/pandas)
        """
        normalizado: Dict[str, float] = {}
        for k, v in raw_scores.items():
            score = float(v)
            if k in fmap:
                # Já veio em formato f{idx}
                normalizado[k] = normalizado.get(k, 0.0) + score
            elif k in feature_set:
                # Veio com nome da coluna; converte para f{idx}
                idx = col_to_idx[k]
                f_id = f"f{idx}"
                normalizado[f_id] = normalizado.get(f_id, 0.0) + score
            else:
                # Ignorar chaves inesperadas
                continue
        return normalizado

    def ordenar_importancia(raw_scores: Dict[str, float], tipo: str) -> List[Dict[str, Any]]:
        scores = normalizar_scores(raw_scores)
        linhas = []
        for f_id, nome in fmap.items():
            score = float(scores.get(f_id, 0.0))
            linhas.append({
                "feature_id": f_id,
                "feature": nome,
                "score": score,
                "type": tipo
            })
        linhas.sort(key=lambda x: x["score"], reverse=True)
        return linhas

    gain_sorted = ordenar_importancia(booster.get_score(importance_type="gain"), "gain")
    weight_sorted = ordenar_importancia(booster.get_score(importance_type="weight"), "weight")
    cover_sorted = ordenar_importancia(booster.get_score(importance_type="cover"), "cover")

    sk_scores = [float(v) for v in clf.feature_importances_]
    sk_sorted = [
        {"feature_id": f"f{i}", "feature": nome, "score": sk_scores[i], "type": "sklearn"}
        for i, nome in enumerate(feature_cols)
    ]
    sk_sorted.sort(key=lambda x: x["score"], reverse=True)

    n = max(1, min(int(top_n), len(feature_cols)))
    return {
        "top_n": n,
        "by_gain_top_n": gain_sorted[:n],
        "by_weight_top_n": weight_sorted[:n],
        "by_cover_top_n": cover_sorted[:n],
        "by_sklearn_top_n": sk_sorted[:n]
    }


def _salvar_graficos_xgboost(clf: xgb.XGBClassifier, feature_importance: Dict[str, Any], reports_dir: Path, logger: logging.Logger) -> None:
    """
    Salva gráficos de treino/validação para diagnóstico visual de underfit/overfit.
    """
    try:
        evals = clf.evals_result()
    except Exception as e:
        logger.warning(f"Nao foi possivel obter evals_result() para graficos: {e}")
        return

    train_hist = evals.get("validation_0", {})
    val_hist = evals.get("validation_1", {})

    for metrica in ("mlogloss", "merror"):
        train_vals = train_hist.get(metrica, [])
        val_vals = val_hist.get(metrica, [])
        if not train_vals or not val_vals:
            continue

        ep = list(range(1, len(train_vals) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(ep, train_vals, label=f"train_{metrica}")
        plt.plot(ep, val_vals, label=f"val_{metrica}")
        plt.xlabel("Iteracao")
        plt.ylabel(metrica)
        plt.title(f"XGBoost - Curva de {metrica}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / f"xgb_curva_{metrica}.png")
        plt.close()

        gap = [v - t for t, v in zip(train_vals, val_vals)]
        plt.figure(figsize=(10, 4))
        plt.plot(ep, gap, label=f"gap_{metrica} (val-train)")
        plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
        plt.xlabel("Iteracao")
        plt.ylabel("Gap")
        plt.title(f"XGBoost - Gap de generalizacao ({metrica})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / f"xgb_gap_{metrica}.png")
        plt.close()

    gain_top = feature_importance.get("by_gain_top_n", [])
    if gain_top:
        labels = [x["feature"] for x in gain_top][::-1]
        scores = [float(x["score"]) for x in gain_top][::-1]
        plt.figure(figsize=(10, 8))
        plt.barh(labels, scores)
        plt.xlabel("Gain")
        plt.title("XGBoost - Importancia de Features (Top por Gain)")
        plt.tight_layout()
        plt.savefig(reports_dir / "xgb_importancia_gain_topn.png")
        plt.close()

    logger.info(f"Graficos de treino XGBoost salvos em: {reports_dir}")


def _treinar_rf(X, y, config_cls, models_dir, reports_dir, logger):
    logger.info("Treinando RandomForest (Baseline)...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    
    # Sem early stopping nativo da mesma forma, usa CV ou OOB
    val_frac = config_cls.get("validacao_interna", {}).get("fracao", 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_frac, random_state=42, stratify=y)
    
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec = recall_score(y_val, preds, average="macro", zero_division=0)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    logger.info(
        f"RF Validação Interna - Acurácia: {acc:.4f}, Precision-Macro: {prec:.4f}, "
        f"Recall-Macro: {rec:.4f}, F1-Macro: {f1:.4f}"
    )
    
    model_path = models_dir / "rf_model.joblib"
    joblib.dump(clf, model_path)
    
    return model_path
