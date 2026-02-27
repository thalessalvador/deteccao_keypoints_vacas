import logging
import json
import joblib
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    elif model_type == "catboost":
        return _treinar_catboost(X, y, groups, origem_instancia, cls_config, models_dir, reports_dir, feature_cols, aug_stats, logger)
    elif model_type == "sklearn_rf":
        return _treinar_rf(X, y, groups, origem_instancia, cls_config, models_dir, reports_dir, logger)
    elif model_type == "svm":
        return _treinar_svm(X, y, groups, origem_instancia, cls_config, models_dir, reports_dir, logger)
    elif model_type == "mlp":
        return _treinar_mlp(X, y, groups, origem_instancia, cls_config, models_dir, reports_dir, logger)
    else:
        logger.warning(f"Modelo {model_type} desconhecido. Usando XGBoost.")
        return _treinar_xgboost(X, y, groups, origem_instancia, cls_config, models_dir, reports_dir, feature_cols, aug_stats, logger)


def _preparar_split_interno_com_grupos(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    origem_instancia: np.ndarray,
    config_cls: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepara split interno por grupos (arquivo), com opcao de validacao apenas em amostras reais.

    Parametros:
        X (pd.DataFrame): Matriz de features completa.
        y (np.ndarray): Vetor de classes.
        groups (np.ndarray): Vetor de grupos para evitar vazamento entre treino/validacao.
        origem_instancia (np.ndarray): Origem da instancia (real/augmentada) por amostra.
        config_cls (Dict[str, Any]): Configuracoes da etapa de classificacao.
        logger (logging.Logger): Logger para mensagens operacionais.

    Retorno:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: X_train, X_val, y_train, y_val.
    """
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
    return X_train, X_val, y_train, y_val

def _treinar_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    origem_instancia: np.ndarray,
    config_cls: Dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
    feature_cols: List[str],
    aug_stats: Dict[str, Any],
    logger: logging.Logger
) -> Path:
    """
    Treina modelo XGBoost com split interno por grupos (arquivo) e early stopping.

    Args:
        X (pd.DataFrame): Matriz de features.
        y (np.ndarray): Vetor de labels codificados.
        groups (np.ndarray): Identificador de grupo por amostra (arquivo).
        origem_instancia (np.ndarray): Origem da amostra ("real" ou "augmentation").
        config_cls (Dict[str, Any]): Configuracoes da secao classificacao.
        models_dir (Path): Diretorio de saida dos modelos.
        reports_dir (Path): Diretorio de saida dos relatorios.
        feature_cols (List[str]): Nomes das features usadas.
        aug_stats (Dict[str, Any]): Resumo de estatisticas de augmentation.
        logger (logging.Logger): Logger configurado.

    Returns:
        Path: Caminho do modelo salvo.
    """
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
    params_base = _obter_parametros_base_xgboost(config_cls, num_class, int(early_stopping_rounds))
    params_finais = dict(params_base)
    resumo_otimizacao: Optional[Dict[str, Any]] = None

    cfg_otimizacao = config_cls.get("otimizacao_hiperparametros", {})
    if bool(cfg_otimizacao.get("habilitar", False)):
        params_otimizados, resumo_otimizacao = _otimizar_hiperparametros_xgboost(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params_base=params_base,
            config_cls=config_cls,
            logger=logger,
        )
        params_finais.update(params_otimizados)
        logger.info(
            f"Otimização finalizada ({resumo_otimizacao.get('metodo_executado', 'desconhecido')}). "
            f"Melhor F1-macro validacao: {resumo_otimizacao.get('melhor_f1_macro', 0.0):.4f}"
        )

    clf = xgb.XGBClassifier(**params_finais)
    
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
        "hiperparametros_usados": params_finais,
        "otimizacao_hiperparametros": resumo_otimizacao,
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


def _obter_parametros_base_xgboost(
    config_cls: Dict[str, Any],
    num_class: int,
    early_stopping_rounds: int,
) -> Dict[str, Any]:
    """
    Monta parametros base do XGBoost a partir do config (com defaults seguros).

    Parametros:
        config_cls (Dict[str, Any]): Configuracoes de classificacao.
        num_class (int): Quantidade de classes do problema.
        early_stopping_rounds (int): Rodadas para early stopping.

    Retorno:
        Dict[str, Any]: Dicionario de parametros base para o XGBoost.
    """
    cfg_xgb = config_cls.get("xgboost", {})
    device_cfg = str(cfg_xgb.get("device", config_cls.get("device", "cpu"))).lower()
    device = "cpu" if device_cfg == "cpu" else "cuda"

    return {
        "n_estimators": int(cfg_xgb.get("n_estimators", 1000)),
        "learning_rate": float(cfg_xgb.get("learning_rate", 0.05)),
        "max_depth": int(cfg_xgb.get("max_depth", 6)),
        "min_child_weight": float(cfg_xgb.get("min_child_weight", 1.0)),
        "subsample": float(cfg_xgb.get("subsample", 0.8)),
        "colsample_bytree": float(cfg_xgb.get("colsample_bytree", 0.8)),
        "gamma": float(cfg_xgb.get("gamma", 0.0)),
        "reg_alpha": float(cfg_xgb.get("reg_alpha", 0.0)),
        "reg_lambda": float(cfg_xgb.get("reg_lambda", 1.0)),
        "objective": "multi:softprob",
        "num_class": int(num_class),
        "eval_metric": ["mlogloss", "merror"],
        "random_state": int(cfg_xgb.get("random_state", 42)),
        "tree_method": str(cfg_xgb.get("tree_method", "hist")),
        "device": device,
        "n_jobs": int(cfg_xgb.get("n_jobs", -1)),
        "early_stopping_rounds": int(early_stopping_rounds),
    }


def _amostrar_parametros_xgboost_random(
    gerador: np.random.Generator,
) -> Dict[str, Any]:
    """
    Gera uma amostra aleatoria de hiperparametros para XGBoost.

    Parametros:
        gerador (np.random.Generator): Gerador pseudoaleatorio.

    Retorno:
        Dict[str, Any]: Hiperparametros amostrados para um trial.
    """
    return {
        "n_estimators": int(gerador.integers(400, 1601)),
        "learning_rate": float(np.exp(gerador.uniform(np.log(0.01), np.log(0.2)))),
        "max_depth": int(gerador.integers(3, 11)),
        "min_child_weight": float(np.exp(gerador.uniform(np.log(0.5), np.log(12.0)))),
        "subsample": float(gerador.uniform(0.6, 1.0)),
        "colsample_bytree": float(gerador.uniform(0.6, 1.0)),
        "gamma": float(gerador.uniform(0.0, 5.0)),
        "reg_alpha": float(np.exp(gerador.uniform(np.log(1e-4), np.log(5.0)))),
        "reg_lambda": float(np.exp(gerador.uniform(np.log(0.1), np.log(20.0)))),
    }


def _avaliar_xgboost_em_validacao(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params_base: Dict[str, Any],
    params_trial: Dict[str, Any],
) -> float:
    """
    Treina um XGBoost com parametros trial e retorna F1-macro na validacao.

    Parametros:
        X_train (pd.DataFrame): Features de treino.
        y_train (np.ndarray): Rotulos de treino.
        X_val (pd.DataFrame): Features de validacao.
        y_val (np.ndarray): Rotulos de validacao.
        params_base (Dict[str, Any]): Parametros base do modelo.
        params_trial (Dict[str, Any]): Parametros candidatos do trial.

    Retorno:
        float: Valor de F1-macro na validacao.
    """
    params = dict(params_base)
    params.update(params_trial)
    clf_trial = xgb.XGBClassifier(**params)
    clf_trial.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    preds_val = clf_trial.predict(X_val)
    return float(f1_score(y_val, preds_val, average="macro", zero_division=0))


def _otimizar_hiperparametros_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params_base: Dict[str, Any],
    config_cls: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Otimiza hiperparametros do XGBoost por Optuna (preferencial) ou random search.

    Parametros:
        X_train (pd.DataFrame): Features de treino.
        y_train (np.ndarray): Rotulos de treino.
        X_val (pd.DataFrame): Features de validacao.
        y_val (np.ndarray): Rotulos de validacao.
        params_base (Dict[str, Any]): Parametros base do modelo.
        config_cls (Dict[str, Any]): Configuracoes da classificacao.
        logger (logging.Logger): Logger para mensagens de progresso.

    Retorno:
        Tuple[Dict[str, Any], Dict[str, Any]]: Melhor conjunto de parametros e resumo da otimizacao.
    """
    cfg_otimizacao = config_cls.get("otimizacao_hiperparametros", {})
    metodo = str(cfg_otimizacao.get("metodo", "optuna")).lower()
    n_trials = int(cfg_otimizacao.get("n_trials", 40))
    seed = int(cfg_otimizacao.get("seed", 42))
    timeout_segundos = cfg_otimizacao.get("timeout_segundos")

    if metodo == "optuna":
        try:
            import optuna  # type: ignore

            logger.info(f"Iniciando otimização de hiperparametros com Optuna ({n_trials} trials)...")
            sampler = optuna.samplers.TPESampler(seed=seed)
            estudo = optuna.create_study(direction="maximize", sampler=sampler)

            def objetivo(trial: Any) -> float:
                """Função objetivo para busca de hiperparâmetros do XGBoost."""
                params_trial = {
                    "n_estimators": trial.suggest_int("n_estimators", 400, 1600),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 12.0, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
                }
                return _avaliar_xgboost_em_validacao(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    params_base=params_base,
                    params_trial=params_trial,
                )

            estudo.optimize(
                objetivo,
                n_trials=n_trials,
                timeout=int(timeout_segundos) if timeout_segundos else None,
                show_progress_bar=False,
            )
            melhor_params = dict(estudo.best_params)
            resumo = {
                "habilitado": True,
                "metodo_solicitado": metodo,
                "metodo_executado": "optuna",
                "n_trials_solicitados": n_trials,
                "n_trials_executados": len(estudo.trials),
                "melhor_f1_macro": float(estudo.best_value),
                "historico_trials": [
                    {"trial": int(t.number) + 1, "value": (None if t.value is None else float(t.value))}
                    for t in estudo.trials
                ],
            }
            return melhor_params, resumo
        except Exception as exc:
            logger.warning(
                f"Falha ao usar Optuna ({exc}). Fallback para random search com {n_trials} trials."
            )

    logger.info(f"Iniciando otimização de hiperparametros com random search ({n_trials} trials)...")
    gerador = np.random.default_rng(seed)
    melhor_score = -1.0
    melhor_params: Dict[str, Any] = {}
    historico_trials: List[Dict[str, Any]] = []

    for _ in range(n_trials):
        params_trial = _amostrar_parametros_xgboost_random(gerador)
        score_trial = _avaliar_xgboost_em_validacao(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params_base=params_base,
            params_trial=params_trial,
        )
        if score_trial > melhor_score:
            melhor_score = score_trial
            melhor_params = params_trial
        historico_trials.append({"trial": len(historico_trials) + 1, "value": float(score_trial)})

    resumo = {
        "habilitado": True,
        "metodo_solicitado": metodo,
        "metodo_executado": "random_search",
        "n_trials_solicitados": n_trials,
        "n_trials_executados": n_trials,
        "melhor_f1_macro": float(melhor_score),
        "historico_trials": historico_trials,
    }
    return melhor_params, resumo


def _treinar_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    origem_instancia: np.ndarray,
    config_cls: Dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
    feature_cols: List[str],
    aug_stats: Dict[str, Any],
    logger: logging.Logger
) -> Path:
    """
    Treina modelo CatBoost com split interno por grupos (arquivo) e early stopping.

    Parametros:
        X (pd.DataFrame): Features completas.
        y (np.ndarray): Rotulos.
        groups (np.ndarray): Grupos para split sem vazamento.
        origem_instancia (np.ndarray): Origem da amostra (real/augmentada).
        config_cls (Dict[str, Any]): Configuracoes de classificacao.
        models_dir (Path): Diretorio para salvar modelos.
        reports_dir (Path): Diretorio para salvar relatorios.
        feature_cols (List[str]): Lista de nomes das features.
        aug_stats (Dict[str, Any]): Estatisticas de augmentacao.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        Path: Caminho do modelo salvo.
    """
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        logger.error("CatBoost não está disponível. Instale com: python -m pip install catboost")
        raise exc

    X_train, X_val, y_train, y_val = _preparar_split_interno_com_grupos(
        X=X,
        y=y,
        groups=groups,
        origem_instancia=origem_instancia,
        config_cls=config_cls,
        logger=logger,
    )

    early_stopping_rounds = config_cls.get("validacao_interna", {}).get("early_stopping_rounds", 50)
    num_class = len(np.unique(y))
    params_base = _obter_parametros_base_catboost(config_cls, num_class)
    params_finais = dict(params_base)
    resumo_otimizacao: Optional[Dict[str, Any]] = None

    cfg_otimizacao = config_cls.get("otimizacao_hiperparametros", {})
    if bool(cfg_otimizacao.get("habilitar", False)):
        params_otimizados, resumo_otimizacao = _otimizar_hiperparametros_catboost(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params_base=params_base,
            config_cls=config_cls,
            early_stopping_rounds=int(early_stopping_rounds),
            logger=logger,
        )
        params_finais.update(params_otimizados)
        logger.info(
            f"Otimização finalizada ({resumo_otimizacao.get('metodo_executado', 'desconhecido')}). "
            f"Melhor F1-macro validacao: {resumo_otimizacao.get('melhor_f1_macro', 0.0):.4f}"
        )

    clf = CatBoostClassifier(**params_finais)
    logger.info("Iniciando treino CatBoost com Early Stopping...")
    clf.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose=False,
    )

    best_iter = int(clf.get_best_iteration())
    logger.info(f"Melhor iteração: {best_iter}")

    preds = clf.predict(X_val)
    preds = np.asarray(preds).reshape(-1).astype(int)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="macro", zero_division=0)
    rec = recall_score(y_val, preds, average="macro", zero_division=0)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    logger.info(
        f"Validação Interna - Acurácia: {acc:.4f}, Precision-Macro: {prec:.4f}, "
        f"Recall-Macro: {rec:.4f}, F1-Macro: {f1:.4f}"
    )

    model_path = models_dir / "catboost_model.cbm"
    clf.save_model(str(model_path))
    logger.info(f"Modelo salvo em {model_path}")
    joblib.dump(feature_cols, models_dir / "feature_names.pkl")

    top_n_imp = config_cls.get("top_n_importancias", 20)
    feature_importance = _extrair_importancia_catboost(clf, feature_cols, top_n=top_n_imp)
    _salvar_grafico_importancia_generico(
        feature_importance=feature_importance,
        chave_top_n="by_catboost_top_n",
        reports_dir=reports_dir,
        nome_arquivo="catboost_importancia_topn.png",
        titulo="CatBoost - Importancia de Features (Top-N)",
        logger=logger,
    )
    _salvar_grafico_otimizacao_generico(
        resumo_otimizacao=resumo_otimizacao,
        reports_dir=reports_dir,
        nome_arquivo="catboost_otimizacao_trials.png",
        titulo="CatBoost - Evolucao dos Trials de Otimizacao",
        logger=logger,
    )

    metrics = {
        "modelo": "catboost",
        "best_iteration": best_iter,
        "hiperparametros_usados": params_finais,
        "otimizacao_hiperparametros": resumo_otimizacao,
        "internal_validation": {
            "accuracy": acc,
            "f1_macro": f1
        },
        "augmentation_stats": aug_stats,
        "feature_importance": feature_importance,
        "classification_report": classification_report(y_val, preds, output_dict=True, zero_division=0)
    }

    with open(reports_dir / "metricas_classificacao_treino.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    return model_path


def _obter_parametros_base_catboost(
    config_cls: Dict[str, Any],
    num_class: int,
) -> Dict[str, Any]:
    """
    Monta parametros base do CatBoost a partir do config (com defaults seguros).

    Parametros:
        config_cls (Dict[str, Any]): Configuracoes de classificacao.
        num_class (int): Quantidade de classes.

    Retorno:
        Dict[str, Any]: Dicionario de parametros base para o CatBoost.
    """
    cfg_cb = config_cls.get("catboost", {})
    device_cfg = str(cfg_cb.get("device", config_cls.get("device", "cpu"))).lower()
    task_type = "CPU" if device_cfg == "cpu" else "GPU"

    params_base: Dict[str, Any] = {
        "iterations": int(cfg_cb.get("iterations", 1000)),
        "learning_rate": float(cfg_cb.get("learning_rate", 0.05)),
        "depth": int(cfg_cb.get("depth", 6)),
        "l2_leaf_reg": float(cfg_cb.get("l2_leaf_reg", 3.0)),
        "random_strength": float(cfg_cb.get("random_strength", 1.0)),
        "bagging_temperature": float(cfg_cb.get("bagging_temperature", 0.5)),
        "loss_function": "MultiClass",
        "eval_metric": str(cfg_cb.get("eval_metric", "MultiClass")),
        "classes_count": int(num_class),
        "random_seed": int(cfg_cb.get("random_seed", 42)),
        "task_type": task_type,
        "verbose": False,
    }
    if task_type == "CPU":
        params_base["rsm"] = float(cfg_cb.get("rsm", 0.8))
    return params_base


def _amostrar_parametros_catboost_random(
    gerador: np.random.Generator,
    usar_rsm: bool,
) -> Dict[str, Any]:
    """
    Gera amostra aleatoria de hiperparametros para CatBoost.

    Parametros:
        gerador (np.random.Generator): Gerador pseudoaleatorio.
        usar_rsm (bool): Indica se o parametro rsm pode ser usado no backend atual.

    Retorno:
        Dict[str, Any]: Parametros candidatos para um trial.
    """
    params_trial: Dict[str, Any] = {
        "iterations": int(gerador.integers(400, 1601)),
        "learning_rate": float(np.exp(gerador.uniform(np.log(0.01), np.log(0.2)))),
        "depth": int(gerador.integers(4, 11)),
        "l2_leaf_reg": float(np.exp(gerador.uniform(np.log(1.0), np.log(20.0)))),
        "random_strength": float(gerador.uniform(0.0, 5.0)),
        "bagging_temperature": float(gerador.uniform(0.0, 3.0)),
    }
    if usar_rsm:
        params_trial["rsm"] = float(gerador.uniform(0.6, 1.0))
    return params_trial


def _avaliar_catboost_em_validacao(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params_base: Dict[str, Any],
    params_trial: Dict[str, Any],
    early_stopping_rounds: int,
) -> float:
    """
    Treina um CatBoost com parametros trial e retorna F1-macro na validacao.

    Parametros:
        X_train (pd.DataFrame): Features de treino.
        y_train (np.ndarray): Rotulos de treino.
        X_val (pd.DataFrame): Features de validacao.
        y_val (np.ndarray): Rotulos de validacao.
        params_base (Dict[str, Any]): Parametros base do modelo.
        params_trial (Dict[str, Any]): Parametros candidatos do trial.
        early_stopping_rounds (int): Rodadas para early stopping.

    Retorno:
        float: F1-macro na validacao.
    """
    from catboost import CatBoostClassifier

    params = dict(params_base)
    params.update(params_trial)
    clf_trial = CatBoostClassifier(**params)
    clf_trial.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose=False,
    )
    preds_val = np.asarray(clf_trial.predict(X_val)).reshape(-1).astype(int)
    return float(f1_score(y_val, preds_val, average="macro", zero_division=0))


def _otimizar_hiperparametros_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params_base: Dict[str, Any],
    config_cls: Dict[str, Any],
    early_stopping_rounds: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Otimiza hiperparametros do CatBoost por Optuna (preferencial) ou random search.

    Parametros:
        X_train (pd.DataFrame): Features de treino.
        y_train (np.ndarray): Rotulos de treino.
        X_val (pd.DataFrame): Features de validacao.
        y_val (np.ndarray): Rotulos de validacao.
        params_base (Dict[str, Any]): Parametros base.
        config_cls (Dict[str, Any]): Configuracoes da classificacao.
        early_stopping_rounds (int): Rodadas para early stopping.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        Tuple[Dict[str, Any], Dict[str, Any]]: Melhor conjunto de parametros e resumo da busca.
    """
    cfg_otimizacao = config_cls.get("otimizacao_hiperparametros", {})
    metodo = str(cfg_otimizacao.get("metodo", "optuna")).lower()
    n_trials = int(cfg_otimizacao.get("n_trials", 40))
    seed = int(cfg_otimizacao.get("seed", 42))
    timeout_segundos = cfg_otimizacao.get("timeout_segundos")
    task_type = str(params_base.get("task_type", "CPU")).upper()
    usar_rsm = task_type == "CPU"
    if not usar_rsm:
        logger.info("CatBoost em GPU detectado: parametro rsm sera ignorado na otimizacao.")

    if metodo == "optuna":
        try:
            import optuna  # type: ignore

            logger.info(f"Iniciando otimização de hiperparametros com Optuna ({n_trials} trials)...")
            sampler = optuna.samplers.TPESampler(seed=seed)
            estudo = optuna.create_study(direction="maximize", sampler=sampler)

            def objetivo(trial: Any) -> float:
                """Função objetivo para busca de hiperparâmetros do CatBoost."""
                params_trial = {
                    "iterations": trial.suggest_int("iterations", 400, 1600),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                    "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
                }
                if usar_rsm:
                    params_trial["rsm"] = trial.suggest_float("rsm", 0.6, 1.0)
                return _avaliar_catboost_em_validacao(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    params_base=params_base,
                    params_trial=params_trial,
                    early_stopping_rounds=int(early_stopping_rounds),
                )

            estudo.optimize(
                objetivo,
                n_trials=n_trials,
                timeout=int(timeout_segundos) if timeout_segundos else None,
                show_progress_bar=False,
            )
            melhor_params = dict(estudo.best_params)
            resumo = {
                "habilitado": True,
                "metodo_solicitado": metodo,
                "metodo_executado": "optuna",
                "n_trials_solicitados": n_trials,
                "n_trials_executados": len(estudo.trials),
                "melhor_f1_macro": float(estudo.best_value),
                "historico_trials": [
                    {"trial": int(t.number) + 1, "value": (None if t.value is None else float(t.value))}
                    for t in estudo.trials
                ],
            }
            return melhor_params, resumo
        except Exception as exc:
            logger.warning(
                f"Falha ao usar Optuna ({exc}). Fallback para random search com {n_trials} trials."
            )

    logger.info(f"Iniciando otimização de hiperparametros com random search ({n_trials} trials)...")
    gerador = np.random.default_rng(seed)
    melhor_score = -1.0
    melhor_params: Dict[str, Any] = {}
    historico_trials: List[Dict[str, Any]] = []

    for _ in range(n_trials):
        params_trial = _amostrar_parametros_catboost_random(gerador, usar_rsm=usar_rsm)
        score_trial = _avaliar_catboost_em_validacao(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params_base=params_base,
            params_trial=params_trial,
            early_stopping_rounds=int(early_stopping_rounds),
        )
        if score_trial > melhor_score:
            melhor_score = score_trial
            melhor_params = params_trial
        historico_trials.append({"trial": len(historico_trials) + 1, "value": float(score_trial)})

    resumo = {
        "habilitado": True,
        "metodo_solicitado": metodo,
        "metodo_executado": "random_search",
        "n_trials_solicitados": n_trials,
        "n_trials_executados": n_trials,
        "melhor_f1_macro": float(melhor_score),
        "historico_trials": historico_trials,
    }
    return melhor_params, resumo


def _extrair_importancia_catboost(
    clf: Any,
    feature_cols: List[str],
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Extrai ranking de importancia do CatBoost.

    Parametros:
        clf (Any): Modelo CatBoost treinado.
        feature_cols (List[str]): Nomes das features.
        top_n (int): Quantidade de features no ranking principal.

    Retorno:
        Dict[str, Any]: Estrutura de importancia no formato do relatorio.
    """
    importancias = [float(v) for v in clf.get_feature_importance()]
    linhas = []
    for i, nome in enumerate(feature_cols):
        linhas.append({
            "feature_id": f"f{i}",
            "feature": nome,
            "score": importancias[i] if i < len(importancias) else 0.0,
            "type": "catboost",
        })
    linhas.sort(key=lambda x: x["score"], reverse=True)

    n = max(1, min(int(top_n), len(feature_cols)))
    return {
        "top_n": n,
        "by_catboost_top_n": linhas[:n],
    }


def _extrair_importancia_rf(
    clf: RandomForestClassifier,
    feature_cols: List[str],
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Extrai ranking de importancia de features do RandomForest.

    Parametros:
        clf (RandomForestClassifier): Modelo RF treinado.
        feature_cols (List[str]): Nomes das features.
        top_n (int): Quantidade de features no ranking principal.

    Retorno:
        Dict[str, Any]: Estrutura de importancia no formato do relatorio.
    """
    scores = [float(v) for v in clf.feature_importances_]
    linhas = [
        {"feature_id": f"f{i}", "feature": nome, "score": scores[i], "type": "rf"}
        for i, nome in enumerate(feature_cols)
    ]
    linhas.sort(key=lambda x: x["score"], reverse=True)
    n = max(1, min(int(top_n), len(feature_cols)))
    return {
        "top_n": n,
        "by_rf_top_n": linhas[:n],
    }


def _salvar_grafico_importancia_generico(
    feature_importance: Dict[str, Any],
    chave_top_n: str,
    reports_dir: Path,
    nome_arquivo: str,
    titulo: str,
    logger: logging.Logger,
) -> None:
    """
    Salva grafico horizontal da importancia de features para modelos tabulares.

    Parametros:
        feature_importance (Dict[str, Any]): Dicionario com ranking de importancia.
        chave_top_n (str): Chave do ranking a ser plotado.
        reports_dir (Path): Diretorio de saida dos graficos.
        nome_arquivo (str): Nome do arquivo de saida.
        titulo (str): Titulo do grafico.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        None: Salva arquivo de grafico em disco.
    """
    linhas = feature_importance.get(chave_top_n, [])
    if not linhas:
        return
    try:
        labels = [x["feature"] for x in linhas][::-1]
        scores = [float(x["score"]) for x in linhas][::-1]
        plt.figure(figsize=(10, 8))
        plt.barh(labels, scores)
        plt.xlabel("Score")
        plt.title(titulo)
        plt.tight_layout()
        plt.savefig(reports_dir / nome_arquivo)
        plt.close()
    except Exception as e:
        logger.warning(f"Nao foi possivel salvar grafico de importancia ({nome_arquivo}): {e}")


def _salvar_grafico_otimizacao_generico(
    resumo_otimizacao: Optional[Dict[str, Any]],
    reports_dir: Path,
    nome_arquivo: str,
    titulo: str,
    logger: logging.Logger,
) -> None:
    """
    Salva curva de evolução de trials quando historico de otimizacao estiver disponivel.

    Parametros:
        resumo_otimizacao (Optional[Dict[str, Any]]): Resumo com historico de trials.
        reports_dir (Path): Diretorio de saida dos graficos.
        nome_arquivo (str): Nome do arquivo de saida.
        titulo (str): Titulo do grafico.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        None: Salva arquivo de grafico em disco.
    """
    if not resumo_otimizacao:
        return
    historico = resumo_otimizacao.get("historico_trials", [])
    if not historico:
        return
    try:
        x = [int(item.get("trial", i + 1)) for i, item in enumerate(historico)]
        y = [float(item.get("value", np.nan)) for item in historico]
        melhor = []
        best_so_far = -np.inf
        for val in y:
            if np.isnan(val):
                melhor.append(best_so_far if best_so_far > -np.inf else np.nan)
            else:
                best_so_far = max(best_so_far, val)
                melhor.append(best_so_far)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="F1 trial", alpha=0.6)
        plt.plot(x, melhor, label="Melhor acumulado", linewidth=2)
        plt.xlabel("Trial")
        plt.ylabel("F1-macro")
        plt.title(titulo)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(reports_dir / nome_arquivo)
        plt.close()
    except Exception as e:
        logger.warning(f"Nao foi possivel salvar grafico de otimizacao ({nome_arquivo}): {e}")


def _resumir_augmentation_em_df(df_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Resume quantidade de instancias reais e augmentadas no dataframe de treino.

    Args:
        df_train (pd.DataFrame): DataFrame de treino.

    Returns:
        Dict[str, Any]: Estatisticas de contagem por origem de instancia.
    """
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

    Parametros:
        clf (xgb.XGBClassifier): Modelo XGBoost treinado.
        feature_cols (List[str]): Lista de nomes das features.
        top_n (int): Quantidade de features no ranking principal.

    Retorno:
        Dict[str, Any]: Dicionario com rankings por diferentes criterios.
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

        Parametros:
            raw_scores (Dict[str, float]): Scores brutos por feature.

        Retorno:
            Dict[str, float]: Scores normalizados para chaves no formato f{idx}.
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
        """
        Monta lista ordenada de importancia no formato padronizado do relatorio.

        Parametros:
            raw_scores (Dict[str, float]): Scores brutos por feature.
            tipo (str): Tipo de importancia (gain, weight, cover, sklearn).

        Retorno:
            List[Dict[str, Any]]: Lista de features ordenadas por score.
        """
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

    Parametros:
        clf (xgb.XGBClassifier): Modelo treinado.
        feature_importance (Dict[str, Any]): Estrutura com importancias de features.
        reports_dir (Path): Diretorio de saida dos graficos.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        None: Salva graficos de treino/validacao e importancia.
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


def _treinar_rf(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    origem_instancia: np.ndarray,
    config_cls: Dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
    logger: logging.Logger
) -> Path:
    """
    Treina RandomForest com split interno por grupos (arquivo).

    Args:
        X (pd.DataFrame): Matriz de features.
        y (np.ndarray): Vetor de labels codificados.
        groups (np.ndarray): Identificador de grupo por amostra (arquivo).
        origem_instancia (np.ndarray): Origem da amostra ("real" ou "augmentation").
        config_cls (Dict[str, Any]): Configuracoes da secao classificacao.
        models_dir (Path): Diretorio de saida dos modelos.
        reports_dir (Path): Diretorio de saida dos relatorios.
        logger (logging.Logger): Logger configurado.

    Returns:
        Path: Caminho do modelo salvo.
    """
    logger.info("Treinando RandomForest...")
    cfg_rf = config_cls.get("rf", {})
    params_base = {
        "n_estimators": int(cfg_rf.get("n_estimators", 400)),
        "max_depth": (None if cfg_rf.get("max_depth", None) is None else int(cfg_rf.get("max_depth"))),
        "min_samples_split": int(cfg_rf.get("min_samples_split", 2)),
        "min_samples_leaf": int(cfg_rf.get("min_samples_leaf", 1)),
        "max_features": cfg_rf.get("max_features", "sqrt"),
        "bootstrap": bool(cfg_rf.get("bootstrap", True)),
        "class_weight": cfg_rf.get("class_weight", None),
        "random_state": int(cfg_rf.get("random_state", 42)),
        "n_jobs": int(cfg_rf.get("n_jobs", -1)),
    }
    
    X_train, X_val, y_train, y_val = _preparar_split_interno_com_grupos(
        X=X,
        y=y,
        groups=groups,
        origem_instancia=origem_instancia,
        config_cls=config_cls,
        logger=logger,
    )
    
    params_finais = dict(params_base)
    cfg_otimizacao = config_cls.get("otimizacao_hiperparametros", {})
    resumo_otimizacao: Optional[Dict[str, Any]] = None
    if bool(cfg_otimizacao.get("habilitar", False)):
        metodo = str(cfg_otimizacao.get("metodo", "optuna")).lower()
        n_trials = int(cfg_otimizacao.get("n_trials", 40))
        seed = int(cfg_otimizacao.get("seed", 42))
        timeout_segundos = cfg_otimizacao.get("timeout_segundos")
        historico_trials: List[Dict[str, Any]] = []
        if metodo == "optuna":
            try:
                import optuna  # type: ignore
                logger.info(f"Iniciando otimização de hiperparametros com Optuna ({n_trials} trials)...")
                sampler = optuna.samplers.TPESampler(seed=seed)
                estudo = optuna.create_study(direction="maximize", sampler=sampler)

                def objetivo(trial: Any) -> float:
                    """Função objetivo para busca de hiperparâmetros do RandomForest."""
                    params_trial = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                        "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(4, 31))),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                    }
                    params = dict(params_base)
                    params.update(params_trial)
                    clf_trial = RandomForestClassifier(**params)
                    clf_trial.fit(X_train, y_train)
                    preds_trial = clf_trial.predict(X_val)
                    return float(f1_score(y_val, preds_trial, average="macro", zero_division=0))

                estudo.optimize(
                    objetivo,
                    n_trials=n_trials,
                    timeout=(int(timeout_segundos) if timeout_segundos else None),
                    show_progress_bar=False,
                )
                params_finais.update(dict(estudo.best_params))
                resumo_otimizacao = {
                    "habilitado": True,
                    "metodo_solicitado": metodo,
                    "metodo_executado": "optuna",
                    "n_trials_solicitados": n_trials,
                    "n_trials_executados": len(estudo.trials),
                    "melhor_f1_macro": float(estudo.best_value),
                    "historico_trials": [
                        {"trial": int(t.number) + 1, "value": (None if t.value is None else float(t.value))}
                        for t in estudo.trials
                    ],
                }
            except Exception as exc:
                logger.warning(f"Falha ao usar Optuna ({exc}). Fallback para random search com {n_trials} trials.")
                metodo = "random"
        if metodo != "optuna":
            logger.info(f"Iniciando otimização de hiperparametros com random search ({n_trials} trials)...")
            gerador = np.random.default_rng(seed)
            melhor_score = -1.0
            melhor_params: Dict[str, Any] = {}
            for _ in range(n_trials):
                params_trial = {
                    "n_estimators": int(gerador.integers(200, 1501)),
                    "max_depth": None if gerador.random() < 0.25 else int(gerador.integers(4, 31)),
                    "min_samples_split": int(gerador.integers(2, 21)),
                    "min_samples_leaf": int(gerador.integers(1, 11)),
                    "max_features": ["sqrt", "log2", None][int(gerador.integers(0, 3))],
                    "bootstrap": bool(gerador.random() < 0.8),
                    "class_weight": ("balanced" if gerador.random() < 0.5 else None),
                }
                params = dict(params_base)
                params.update(params_trial)
                clf_trial = RandomForestClassifier(**params)
                clf_trial.fit(X_train, y_train)
                preds_trial = clf_trial.predict(X_val)
                score = float(f1_score(y_val, preds_trial, average="macro", zero_division=0))
                if score > melhor_score:
                    melhor_score = score
                    melhor_params = params_trial
                historico_trials.append({"trial": len(historico_trials) + 1, "value": float(score)})
            params_finais.update(melhor_params)
            resumo_otimizacao = {
                "habilitado": True,
                "metodo_solicitado": str(cfg_otimizacao.get("metodo", "random")),
                "metodo_executado": "random_search",
                "n_trials_solicitados": n_trials,
                "n_trials_executados": n_trials,
                "melhor_f1_macro": float(melhor_score),
                "historico_trials": historico_trials,
            }
        if resumo_otimizacao is not None:
            logger.info(
                f"Otimização finalizada ({resumo_otimizacao.get('metodo_executado', 'desconhecido')}). "
                f"Melhor F1-macro validacao: {resumo_otimizacao.get('melhor_f1_macro', 0.0):.4f}"
            )

    clf = RandomForestClassifier(**params_finais)
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
    joblib.dump(list(X.columns), models_dir / "feature_names.pkl")
    logger.info(f"Modelo salvo em {model_path}")

    feature_importance = _extrair_importancia_rf(clf, list(X.columns), top_n=config_cls.get("top_n_importancias", 20))
    _salvar_grafico_importancia_generico(
        feature_importance=feature_importance,
        chave_top_n="by_rf_top_n",
        reports_dir=reports_dir,
        nome_arquivo="rf_importancia_topn.png",
        titulo="RandomForest - Importancia de Features (Top-N)",
        logger=logger,
    )
    _salvar_grafico_otimizacao_generico(
        resumo_otimizacao=resumo_otimizacao,
        reports_dir=reports_dir,
        nome_arquivo="rf_otimizacao_trials.png",
        titulo="RandomForest - Evolucao dos Trials de Otimizacao",
        logger=logger,
    )

    metrics = {
        "modelo": "sklearn_rf",
        "hiperparametros_usados": params_finais,
        "otimizacao_hiperparametros": resumo_otimizacao,
        "internal_validation": {
            "accuracy": acc,
            "f1_macro": f1
        },
        "feature_importance": feature_importance,
        "classification_report": classification_report(y_val, preds, output_dict=True, zero_division=0)
    }
    with open(reports_dir / "metricas_classificacao_treino.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    return model_path


def _treinar_svm(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    origem_instancia: np.ndarray,
    config_cls: Dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
) -> Path:
    """
    Treina SVM (SVC) com split interno por grupos.
    Usa StandardScaler no pipeline e suporta Optuna/random search.

    Parametros:
        X (pd.DataFrame): Features completas.
        y (np.ndarray): Rotulos.
        groups (np.ndarray): Grupos para split sem vazamento.
        origem_instancia (np.ndarray): Origem da amostra (real/augmentada).
        config_cls (Dict[str, Any]): Configuracoes da classificacao.
        models_dir (Path): Diretorio de saida de modelos.
        reports_dir (Path): Diretorio de saida de relatorios.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        Path: Caminho do modelo salvo.
    """
    logger.info("Treinando SVM...")
    cfg_svm = config_cls.get("svm", {})
    params_base = {
        "C": float(cfg_svm.get("C", 1.0)),
        "kernel": str(cfg_svm.get("kernel", "rbf")),
        "gamma": cfg_svm.get("gamma", "scale"),
        "degree": int(cfg_svm.get("degree", 3)),
        "class_weight": cfg_svm.get("class_weight", None),
        "probability": True,
        "random_state": int(cfg_svm.get("random_state", 42)),
    }

    X_train, X_val, y_train, y_val = _preparar_split_interno_com_grupos(
        X=X,
        y=y,
        groups=groups,
        origem_instancia=origem_instancia,
        config_cls=config_cls,
        logger=logger,
    )

    params_finais = dict(params_base)
    cfg_otimizacao = config_cls.get("otimizacao_hiperparametros", {})
    resumo_otimizacao: Optional[Dict[str, Any]] = None
    if bool(cfg_otimizacao.get("habilitar", False)):
        metodo = str(cfg_otimizacao.get("metodo", "optuna")).lower()
        n_trials = int(cfg_otimizacao.get("n_trials", 40))
        seed = int(cfg_otimizacao.get("seed", 42))
        timeout_segundos = cfg_otimizacao.get("timeout_segundos")
        historico_trials: List[Dict[str, Any]] = []

        def _avaliar_trial(params_trial: Dict[str, Any]) -> float:
            params_local = dict(params_base)
            params_local.update(params_trial)
            clf_trial = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("svm", SVC(**params_local)),
                ]
            )
            clf_trial.fit(X_train, y_train)
            preds_val = clf_trial.predict(X_val)
            return float(f1_score(y_val, preds_val, average="macro", zero_division=0))

        if metodo == "optuna":
            try:
                import optuna  # type: ignore
                logger.info(f"Iniciando otimização de hiperparâmetros SVM com Optuna ({n_trials} trials)...")
                sampler = optuna.samplers.TPESampler(seed=seed)
                estudo = optuna.create_study(direction="maximize", sampler=sampler)

                def objetivo(trial: Any) -> float:
                    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
                    params_trial: Dict[str, Any] = {
                        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
                        "kernel": kernel,
                        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
                    }
                    if kernel in ("rbf", "poly"):
                        params_trial["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
                    if kernel == "poly":
                        params_trial["degree"] = trial.suggest_int("degree", 2, 4)
                    return _avaliar_trial(params_trial)

                estudo.optimize(
                    objetivo,
                    n_trials=n_trials,
                    timeout=(int(timeout_segundos) if timeout_segundos else None),
                    show_progress_bar=False,
                )
                params_finais.update(dict(estudo.best_params))
                resumo_otimizacao = {
                    "habilitado": True,
                    "metodo_solicitado": metodo,
                    "metodo_executado": "optuna",
                    "n_trials_solicitados": n_trials,
                    "n_trials_executados": len(estudo.trials),
                    "melhor_f1_macro": float(estudo.best_value),
                    "historico_trials": [
                        {"trial": int(t.number) + 1, "value": (None if t.value is None else float(t.value))}
                        for t in estudo.trials
                    ],
                }
            except Exception as exc:
                logger.warning(f"Falha ao usar Optuna para SVM ({exc}). Fallback para random search.")
                metodo = "random"

        if metodo != "optuna":
            logger.info(f"Iniciando otimização SVM com random search ({n_trials} trials)...")
            gerador = np.random.default_rng(seed)
            melhor_score = -1.0
            melhor_params: Dict[str, Any] = {}
            for _ in range(n_trials):
                kernel = ["rbf", "linear", "poly"][int(gerador.integers(0, 3))]
                params_trial: Dict[str, Any] = {
                    "C": float(np.exp(gerador.uniform(np.log(1e-2), np.log(1e3)))),
                    "kernel": kernel,
                    "class_weight": ("balanced" if gerador.random() < 0.5 else None),
                }
                if kernel in ("rbf", "poly"):
                    params_trial["gamma"] = float(np.exp(gerador.uniform(np.log(1e-4), np.log(1.0))))
                if kernel == "poly":
                    params_trial["degree"] = int(gerador.integers(2, 5))
                score = _avaliar_trial(params_trial)
                if score > melhor_score:
                    melhor_score = score
                    melhor_params = params_trial
                historico_trials.append({"trial": len(historico_trials) + 1, "value": float(score)})
            params_finais.update(melhor_params)
            resumo_otimizacao = {
                "habilitado": True,
                "metodo_solicitado": str(cfg_otimizacao.get("metodo", "random")),
                "metodo_executado": "random_search",
                "n_trials_solicitados": n_trials,
                "n_trials_executados": n_trials,
                "melhor_f1_macro": float(melhor_score),
                "historico_trials": historico_trials,
            }

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(**params_finais)),
        ]
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    logger.info(f"SVM Validação Interna - Acurácia: {acc:.4f}, F1-Macro: {f1:.4f}")

    model_path = models_dir / "svm_model.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(list(X.columns), models_dir / "feature_names.pkl")
    logger.info(f"Modelo salvo em {model_path}")

    _salvar_grafico_otimizacao_generico(
        resumo_otimizacao=resumo_otimizacao,
        reports_dir=reports_dir,
        nome_arquivo="svm_otimizacao_trials.png",
        titulo="SVM - Evolucao dos Trials de Otimizacao",
        logger=logger,
    )

    metrics = {
        "modelo": "svm",
        "hiperparametros_usados": params_finais,
        "otimizacao_hiperparametros": resumo_otimizacao,
        "internal_validation": {"accuracy": acc, "f1_macro": f1},
        "classification_report": classification_report(y_val, preds, output_dict=True, zero_division=0),
    }
    with open(reports_dir / "metricas_classificacao_treino.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return model_path


def _treinar_mlp(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    origem_instancia: np.ndarray,
    config_cls: Dict[str, Any],
    models_dir: Path,
    reports_dir: Path,
    logger: logging.Logger,
) -> Path:
    """
    Treina MLP (scikit-learn) com arquitetura flexível.
    Dimensão de entrada e classes são inferidas automaticamente de X e y.

    Parametros:
        X (pd.DataFrame): Features completas.
        y (np.ndarray): Rotulos.
        groups (np.ndarray): Grupos para split sem vazamento.
        origem_instancia (np.ndarray): Origem da amostra (real/augmentada).
        config_cls (Dict[str, Any]): Configuracoes da classificacao.
        models_dir (Path): Diretorio de saida de modelos.
        reports_dir (Path): Diretorio de saida de relatorios.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        Path: Caminho do modelo salvo.
    """
    logger.info("Treinando MLP...")
    cfg_mlp = config_cls.get("mlp", {})
    n_features = int(X.shape[1])
    n_classes = int(len(np.unique(y)))

    hidden_layers_cfg = cfg_mlp.get("hidden_layer_sizes", [128, 64])
    if isinstance(hidden_layers_cfg, list):
        hidden_layers = tuple(int(v) for v in hidden_layers_cfg)
    else:
        hidden_layers = (128, 64)

    params_base = {
        "hidden_layer_sizes": hidden_layers,
        "activation": str(cfg_mlp.get("activation", "relu")),
        "alpha": float(cfg_mlp.get("alpha", 1e-4)),
        "learning_rate_init": float(cfg_mlp.get("learning_rate_init", 1e-3)),
        "batch_size": int(cfg_mlp.get("batch_size", 32)),
        "max_iter": int(cfg_mlp.get("max_iter", 400)),
        "early_stopping": bool(cfg_mlp.get("early_stopping", True)),
        "validation_fraction": float(cfg_mlp.get("validation_fraction", 0.1)),
        "n_iter_no_change": int(cfg_mlp.get("n_iter_no_change", 30)),
        "random_state": int(cfg_mlp.get("random_state", 42)),
    }
    logger.info(f"MLP dinâmico - n_features: {n_features} | n_classes: {n_classes}")

    X_train, X_val, y_train, y_val = _preparar_split_interno_com_grupos(
        X=X,
        y=y,
        groups=groups,
        origem_instancia=origem_instancia,
        config_cls=config_cls,
        logger=logger,
    )

    params_finais = dict(params_base)
    cfg_otimizacao = config_cls.get("otimizacao_hiperparametros", {})
    resumo_otimizacao: Optional[Dict[str, Any]] = None
    if bool(cfg_otimizacao.get("habilitar", False)):
        metodo = str(cfg_otimizacao.get("metodo", "optuna")).lower()
        n_trials = int(cfg_otimizacao.get("n_trials", 40))
        seed = int(cfg_otimizacao.get("seed", 42))
        timeout_segundos = cfg_otimizacao.get("timeout_segundos")
        historico_trials: List[Dict[str, Any]] = []

        def _avaliar_trial(params_trial: Dict[str, Any]) -> float:
            params_local = dict(params_base)
            params_local.update(params_trial)
            clf_trial = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(**params_local)),
                ]
            )
            clf_trial.fit(X_train, y_train)
            preds_val = clf_trial.predict(X_val)
            return float(f1_score(y_val, preds_val, average="macro", zero_division=0))

        if metodo == "optuna":
            try:
                import optuna  # type: ignore
                logger.info(f"Iniciando otimização de hiperparâmetros MLP com Optuna ({n_trials} trials)...")
                sampler = optuna.samplers.TPESampler(seed=seed)
                estudo = optuna.create_study(direction="maximize", sampler=sampler)

                def objetivo(trial: Any) -> float:
                    n_layers = trial.suggest_int("n_layers", 1, 3)
                    layer_choices = [64, 128, 256]
                    hidden = tuple(
                        trial.suggest_categorical(f"hidden_{i}", layer_choices) for i in range(n_layers)
                    )
                    params_trial = {
                        "hidden_layer_sizes": hidden,
                        "alpha": trial.suggest_float("alpha", 1e-6, 1e-3, log=True),
                        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 3e-3, log=True),
                        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    }
                    return _avaliar_trial(params_trial)

                estudo.optimize(
                    objetivo,
                    n_trials=n_trials,
                    timeout=(int(timeout_segundos) if timeout_segundos else None),
                    show_progress_bar=False,
                )
                best = estudo.best_trial.params
                n_layers_best = int(best.get("n_layers", 1))
                hidden_best = tuple(int(best[f"hidden_{i}"]) for i in range(n_layers_best))
                params_finais.update(
                    {
                        "hidden_layer_sizes": hidden_best,
                        "alpha": float(best.get("alpha", params_base["alpha"])),
                        "learning_rate_init": float(best.get("learning_rate_init", params_base["learning_rate_init"])),
                        "batch_size": int(best.get("batch_size", params_base["batch_size"])),
                    }
                )
                resumo_otimizacao = {
                    "habilitado": True,
                    "metodo_solicitado": metodo,
                    "metodo_executado": "optuna",
                    "n_trials_solicitados": n_trials,
                    "n_trials_executados": len(estudo.trials),
                    "melhor_f1_macro": float(estudo.best_value),
                    "historico_trials": [
                        {"trial": int(t.number) + 1, "value": (None if t.value is None else float(t.value))}
                        for t in estudo.trials
                    ],
                }
            except Exception as exc:
                logger.warning(f"Falha ao usar Optuna para MLP ({exc}). Fallback para random search.")
                metodo = "random"

        if metodo != "optuna":
            logger.info(f"Iniciando otimização MLP com random search ({n_trials} trials)...")
            gerador = np.random.default_rng(seed)
            melhor_score = -1.0
            melhor_params: Dict[str, Any] = {}
            layer_choices = [64, 128, 256]
            for _ in range(n_trials):
                n_layers = int(gerador.integers(1, 4))
                hidden = tuple(layer_choices[int(gerador.integers(0, len(layer_choices)))] for _ in range(n_layers))
                params_trial = {
                    "hidden_layer_sizes": hidden,
                    "alpha": float(np.exp(gerador.uniform(np.log(1e-6), np.log(1e-3)))),
                    "learning_rate_init": float(np.exp(gerador.uniform(np.log(1e-4), np.log(3e-3)))),
                    "batch_size": [16, 32, 64][int(gerador.integers(0, 3))],
                }
                score = _avaliar_trial(params_trial)
                if score > melhor_score:
                    melhor_score = score
                    melhor_params = params_trial
                historico_trials.append({"trial": len(historico_trials) + 1, "value": float(score)})
            params_finais.update(melhor_params)
            resumo_otimizacao = {
                "habilitado": True,
                "metodo_solicitado": str(cfg_otimizacao.get("metodo", "random")),
                "metodo_executado": "random_search",
                "n_trials_solicitados": n_trials,
                "n_trials_executados": n_trials,
                "melhor_f1_macro": float(melhor_score),
                "historico_trials": historico_trials,
            }

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**params_finais)),
        ]
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    hidden_eff = tuple(int(v) for v in params_finais.get("hidden_layer_sizes", hidden_layers))
    arquitetura = [n_features] + list(hidden_eff) + [n_classes]
    total_parametros = 0
    for i in range(len(arquitetura) - 1):
        n_in = int(arquitetura[i])
        n_out = int(arquitetura[i + 1])
        total_parametros += (n_in + 1) * n_out  # pesos + bias
    logger.info(
        "MLP Summary -> arquitetura=%s | ativacao=%s | alpha=%.6f | lr=%.6f | batch_size=%s | max_iter=%s | params=%d",
        arquitetura,
        str(params_finais.get("activation", params_base["activation"])),
        float(params_finais.get("alpha", params_base["alpha"])),
        float(params_finais.get("learning_rate_init", params_base["learning_rate_init"])),
        str(params_finais.get("batch_size", params_base["batch_size"])),
        str(params_finais.get("max_iter", params_base["max_iter"])),
        int(total_parametros),
    )
    logger.info(f"MLP Validação Interna - Acurácia: {acc:.4f}, F1-Macro: {f1:.4f}")

    _salvar_graficos_treino_mlp(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params_finais=params_finais,
        reports_dir=reports_dir,
        logger=logger,
    )

    model_path = models_dir / "mlp_model.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(list(X.columns), models_dir / "feature_names.pkl")
    logger.info(f"Modelo salvo em {model_path}")

    _salvar_grafico_otimizacao_generico(
        resumo_otimizacao=resumo_otimizacao,
        reports_dir=reports_dir,
        nome_arquivo="mlp_otimizacao_trials.png",
        titulo="MLP - Evolucao dos Trials de Otimizacao",
        logger=logger,
    )

    metrics = {
        "modelo": "mlp",
        "hiperparametros_usados": params_finais,
        "otimizacao_hiperparametros": resumo_otimizacao,
        "internal_validation": {"accuracy": acc, "f1_macro": f1},
        "rede_dinamica": {
            "n_features_entrada": n_features,
            "n_classes": n_classes,
            "hidden_layer_sizes": list(params_finais.get("hidden_layer_sizes", hidden_layers)),
        },
        "classification_report": classification_report(y_val, preds, output_dict=True, zero_division=0),
    }
    with open(reports_dir / "metricas_classificacao_treino.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return model_path


def _salvar_graficos_treino_mlp(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params_finais: Dict[str, Any],
    reports_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Salva curvas de loss e acuracia (treino e validacao) para o MLP final.

    Parametros:
        X_train (pd.DataFrame): Features de treino.
        y_train (np.ndarray): Rotulos de treino.
        X_val (pd.DataFrame): Features de validacao interna.
        y_val (np.ndarray): Rotulos de validacao interna.
        params_finais (Dict[str, Any]): Hiperparametros finais do MLP.
        reports_dir (Path): Diretorio de saida dos graficos.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        None: Salva os arquivos de grafico em disco.
    """
    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        params_curva = dict(params_finais)
        max_iter = int(params_curva.get("max_iter", 200))
        params_curva["max_iter"] = 1
        params_curva["warm_start"] = True
        params_curva["early_stopping"] = False

        clf_curva = MLPClassifier(**params_curva)
        classes_all = np.unique(y_train)

        loss_treino: List[float] = []
        loss_validacao: List[float] = []
        acc_treino: List[float] = []
        acc_validacao: List[float] = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for _ in range(max_iter):
                clf_curva.fit(X_train_s, y_train)

                loss_treino.append(float(getattr(clf_curva, "loss_", np.nan)))

                pred_train = clf_curva.predict(X_train_s)
                pred_val = clf_curva.predict(X_val_s)
                acc_treino.append(float(accuracy_score(y_train, pred_train)))
                acc_validacao.append(float(accuracy_score(y_val, pred_val)))

                try:
                    proba_val = clf_curva.predict_proba(X_val_s)
                    loss_val = float(log_loss(y_val, proba_val, labels=classes_all))
                except Exception:
                    loss_val = float("nan")
                loss_validacao.append(loss_val)

        epocas = np.arange(1, max_iter + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epocas, loss_treino, label="Loss treino")
        plt.plot(epocas, loss_validacao, label="Loss validacao")
        plt.xlabel("Epoca")
        plt.ylabel("Loss")
        plt.title("MLP - Evolucao da Loss (Treino vs Validacao)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(reports_dir / "mlp_curva_loss_treino_validacao.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epocas, acc_treino, label="Acuracia treino")
        plt.plot(epocas, acc_validacao, label="Acuracia validacao")
        plt.xlabel("Epoca")
        plt.ylabel("Acuracia")
        plt.title("MLP - Evolucao da Acuracia (Treino vs Validacao)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(reports_dir / "mlp_curva_acuracia_treino_validacao.png")
        plt.close()

        logger.info(
            "Graficos do MLP final salvos em %s: mlp_curva_loss_treino_validacao.png e mlp_curva_acuracia_treino_validacao.png",
            str(reports_dir),
        )
    except Exception as exc:
        logger.warning(f"Nao foi possivel salvar graficos de treino do MLP: {exc}")
