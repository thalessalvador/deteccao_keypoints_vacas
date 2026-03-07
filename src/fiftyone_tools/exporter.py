import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import fiftyone as fo
except Exception:  # pragma: no cover - dependencia opcional
    fo = None

from ..classificacao.mlp_torch import carregar_checkpoint_mlp_torch, predict_proba_mlp_torch
from ..classificacao.siamese_torch import carregar_checkpoint_siamese_torch, predict_proba_siamese_torch

KP_LABELS_POSE = [
    "withers",
    "back",
    "hook_up",
    "hook_down",
    "hip",
    "tail_head",
    "pin_up",
    "pin_down",
]

# Paleta viva com bom contraste em vacas pretas/brancas/marrons
KP_COLOR_POOL = [
    "#00FFFF",  # ciano
    "#FF00FF",  # magenta
    "#FFFF00",  # amarelo
    "#00FF00",  # verde-lima
    "#FF4D00",  # laranja forte
    "#39FF14",  # neon green
    "#00A2FF",  # azul vivo
    "#FF1493",  # rosa forte
]


def exportar_para_fiftyone(config: Dict[str, Any], modo: str, launch: bool, logger: logging.Logger) -> None:
    """
    Exporta dados do projeto para datasets no FiftyOne.

    O metodo suporta tres analises principais:
    1) classificacao_teste: imagens do split de teste com predito x real e top-k;
    2) classificacao_raw: catalogo do dataset bruto de classificacao por pasta/classe;
    3) pose_anotacoes: imagens anotadas em formato YOLO Pose (bbox + keypoints).

    Parametros:
        config (Dict[str, Any]): Configuracao global do projeto.
        modo (str): Modo de exportacao. Valores aceitos:
            "classificacao-teste", "classificacao-raw", "pose-anotacoes", "todos".
        launch (bool): Se True, abre o app do FiftyOne ao final.
        logger (logging.Logger): Logger para mensagens de execucao.

    Retorno:
        None: Cria/atualiza datasets no FiftyOne e opcionalmente inicia a interface.
    """
    if fo is None:
        raise RuntimeError(
            "FiftyOne nao esta disponivel neste ambiente. Instale com: pip install fiftyone"
        )

    modo_norm = str(modo).strip().lower()
    modos_validos = {"classificacao-teste", "classificacao-raw", "pose-anotacoes", "todos"}
    if modo_norm not in modos_validos:
        raise ValueError(f"Modo invalido '{modo}'. Use um de: {sorted(modos_validos)}")

    datasets_criados: List[str] = []

    if modo_norm in {"classificacao-teste", "todos"}:
        nome = _exportar_classificacao_teste(config, logger)
        datasets_criados.append(nome)

    if modo_norm in {"classificacao-raw", "todos"}:
        nome = _exportar_classificacao_raw(config, logger)
        datasets_criados.append(nome)

    if modo_norm in {"pose-anotacoes", "todos"}:
        nome = _exportar_pose_anotacoes(config, logger)
        datasets_criados.append(nome)

    logger.info("Datasets exportados para FiftyOne: %s", datasets_criados)

    if launch and datasets_criados:
        ds = fo.load_dataset(datasets_criados[0])
        session = fo.launch_app(ds, auto=False)
        logger.info("FiftyOne aberto em: %s", session.url)
        logger.info("Sessao FiftyOne ativa. Pressione Ctrl+C para encerrar.")
        try:
            session.wait()
        except KeyboardInterrupt:
            logger.info("Encerrando sessao FiftyOne...")
        finally:
            try:
                session.close()
            except Exception:
                pass


def _exportar_classificacao_raw(config: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Exporta o dataset bruto de classificacao para auditoria visual por pasta/classe.

    Cada amostra inclui:
    - caminho da imagem;
    - classe esperada (nome da pasta);
    - nome do arquivo.

    Parametros:
        config (Dict[str, Any]): Configuracao global.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        str: Nome do dataset FiftyOne criado/atualizado.
    """
    raw_cls_dir = Path(config["paths"]["raw"]) / "dataset_classificacao"
    if not raw_cls_dir.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {raw_cls_dir}")

    dataset_name = "vacas_classificacao_raw_auditoria"
    _reset_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    amostras = 0
    for pasta_classe in sorted([p for p in raw_cls_dir.iterdir() if p.is_dir()]):
        classe = pasta_classe.name
        for img_path in sorted(pasta_classe.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            sample = fo.Sample(filepath=str(img_path))
            sample["classe_esperada"] = fo.Classification(label=classe)
            sample["arquivo"] = img_path.name
            sample["pasta_origem"] = str(pasta_classe)
            dataset.add_sample(sample)
            amostras += 1

    logger.info("FiftyOne RAW classificacao: %d amostras exportadas para '%s'", amostras, dataset_name)
    return dataset_name


def _exportar_classificacao_teste(config: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Exporta o conjunto de teste da classificacao com predito x real e top-k.

    O metodo carrega o modelo treinado atual e calcula previsoes para as imagens
    reais do split de teste, registrando campos uteis para investigacao de erros.

    Parametros:
        config (Dict[str, Any]): Configuracao global.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        str: Nome do dataset FiftyOne criado/atualizado.
    """
    processed_dir = Path(config["paths"]["processed"]) / "classificacao"
    features_csv = processed_dir / "features" / "features_completas.csv"
    test_txt = processed_dir / "splits" / "teste_10pct.txt"
    raw_cls_dir = Path(config["paths"]["raw"]) / "dataset_classificacao"
    models_dir = Path(config["paths"]["models"]) / "classificacao" / "modelos_salvos"

    if not features_csv.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {features_csv}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {test_txt}")

    df = pd.read_csv(features_csv)
    with open(test_txt, "r", encoding="utf-8") as f:
        test_files = {line.strip() for line in f if line.strip()}

    df_test = df[df["arquivo"].isin(test_files)].copy()
    if "origem_instancia" in df_test.columns:
        df_test = df_test[df_test["origem_instancia"] == "real"].copy()
    if df_test.empty:
        raise ValueError("Split de teste vazio apos filtro em features_completas.csv")

    feature_names_path = models_dir / "feature_names.pkl"
    label_encoder_path = models_dir / "label_encoder.pkl"
    if not feature_names_path.exists() or not label_encoder_path.exists():
        raise FileNotFoundError("Artefatos do classificador ausentes (feature_names.pkl/label_encoder.pkl)")

    feature_cols = joblib.load(feature_names_path)
    le = joblib.load(label_encoder_path)

    X_test = df_test[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y_true_str = df_test["classe"].astype(str).values
    y_true = le.transform(y_true_str)

    probs = _predict_proba_classificador_atual(config, X_test, models_dir, logger)
    preds = np.argmax(probs, axis=1)

    dataset_name = "vacas_classificacao_teste_predicoes"
    _reset_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    top_k_default = 5
    cache_paths: Dict[str, Optional[Path]] = {}
    amostras = 0
    erros = 0

    for idx, row in df_test.reset_index(drop=True).iterrows():
        arquivo = str(row["arquivo"])
        classe_real = str(row["classe"])
        imagem_path = _resolver_imagem_classificacao(raw_cls_dir, classe_real, arquivo, cache_paths)
        if imagem_path is None:
            # Mantem robustez: pula amostra sem arquivo fisico.
            continue

        pred_idx = int(preds[idx])
        pred_label = str(le.inverse_transform([pred_idx])[0])
        conf = float(probs[idx, pred_idx])
        correto = bool(pred_idx == int(y_true[idx]))
        if not correto:
            erros += 1

        top_idx = np.argsort(probs[idx])[::-1][: min(top_k_default, probs.shape[1])]
        top_k = [
            {
                "classe": str(le.inverse_transform([int(i)])[0]),
                "confianca": float(probs[idx, int(i)]),
            }
            for i in top_idx
        ]

        sample = fo.Sample(filepath=str(imagem_path))
        sample["arquivo"] = arquivo
        sample["ground_truth"] = fo.Classification(label=classe_real)
        sample["prediction"] = fo.Classification(label=pred_label, confidence=conf)
        sample["correto"] = correto
        sample["confianca_top1"] = conf
        sample["top_k_json"] = json.dumps(top_k, ensure_ascii=False)
        sample["classe_real"] = classe_real
        sample["classe_predita"] = pred_label
        dataset.add_sample(sample)
        amostras += 1

    logger.info(
        "FiftyOne teste classificacao: %d amostras exportadas (%d erros) para '%s'",
        amostras,
        erros,
        dataset_name,
    )
    return dataset_name


def _exportar_pose_anotacoes(config: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Exporta anotacoes de pose do dataset YOLO processado para auditoria visual.

    Cada amostra recebe detections (bbox) e keypoints conforme labels YOLO.

    Parametros:
        config (Dict[str, Any]): Configuracao global.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        str: Nome do dataset FiftyOne criado/atualizado.
    """
    yolo_dir = Path(config["paths"]["processed"]) / "yolo_pose"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Diretorio yolo_pose/images ou yolo_pose/labels nao encontrado")

    dataset_name = "vacas_pose_anotacoes_yolo"
    _reset_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)
    dataset.default_skeleton = fo.KeypointSkeleton(
        labels=KP_LABELS_POSE,
        edges=[
            [0, 1],  # withers-back
            [1, 2],  # back-hook_up
            [1, 3],  # back-hook_down
            [1, 4],  # back-hip
            [4, 5],  # hip-tail_head
            [5, 6],  # tail_head-pin_up
            [5, 7],  # tail_head-pin_down
            [2, 6],  # hook_up-pin_up
            [3, 7],  # hook_down-pin_down
        ],
    )
    dataset.app_config.multicolor_keypoints = True
    dataset.app_config.show_skeletons = True
    dataset.app_config.show_label = True
    dataset.app_config.color_pool = KP_COLOR_POOL
    dataset.save()

    amostras = 0
    total_instancias = 0

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        detections: List[Any] = []
        keypoints_por_instancia: List[Any] = []
        keypoints_rotulados_flat: List[Any] = []
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                linhas = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            linhas = []

        for ln in linhas:
            vals = ln.split()
            if len(vals) < 5:
                continue
            try:
                cls_id = int(float(vals[0]))
                xc, yc, w, h = map(float, vals[1:5])
            except Exception:
                continue

            x = max(0.0, xc - w / 2.0)
            y = max(0.0, yc - h / 2.0)
            bbox = [x, y, min(1.0, w), min(1.0, h)]

            kp_vals = vals[5:]
            keypoints_det: List[Any] = []
            pontos_instancia: List[List[float]] = []
            for i in range(0, len(kp_vals), 3):
                if i + 2 >= len(kp_vals):
                    break
                try:
                    kx = float(kp_vals[i])
                    ky = float(kp_vals[i + 1])
                    kv = float(kp_vals[i + 2])
                except Exception:
                    continue
                if kv > 0:
                    pontos_instancia.append([kx, ky])
                    nome_kp = KP_LABELS_POSE[i // 3] if (i // 3) < len(KP_LABELS_POSE) else f"kp_{i // 3}"
                    # Campo de keypoints interno da detection (instancia da vaca)
                    keypoints_det.append(fo.Keypoint(points=[[kx, ky]], label=nome_kp))
                    # Campo "flat" para exibir rotulo de cada keypoint na imagem
                    keypoints_rotulados_flat.append(fo.Keypoint(points=[[kx, ky]], label=nome_kp))

            detections.append(
                fo.Detection(
                    label=f"vaca_{cls_id}",
                    bounding_box=bbox,
                    confidence=1.0,
                    keypoints=keypoints_det,
                )
            )
            if pontos_instancia:
                keypoints_por_instancia.append(
                    fo.Keypoint(points=pontos_instancia, label=f"vaca_{cls_id}")
                )

        sample = fo.Sample(filepath=str(img_path))
        sample["detections"] = fo.Detections(detections=detections)
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints_por_instancia)
        sample["keypoints_rotulados"] = fo.Keypoints(keypoints=keypoints_rotulados_flat)
        sample["num_instancias"] = len(detections)
        dataset.add_sample(sample)
        amostras += 1
        total_instancias += len(detections)

    logger.info(
        "FiftyOne pose anotacoes: %d imagens e %d instancias exportadas para '%s'",
        amostras,
        total_instancias,
        dataset_name,
    )
    return dataset_name


def _predict_proba_classificador_atual(
    config: Dict[str, Any],
    X: pd.DataFrame,
    models_dir: Path,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Calcula probabilidades para o classificador configurado em `modelo_padrao`.

    Parametros:
        config (Dict[str, Any]): Configuracao global.
        X (pd.DataFrame): Features de entrada.
        models_dir (Path): Diretorio com artefatos do classificador.
        logger (logging.Logger): Logger para mensagens.

    Retorno:
        np.ndarray: Matriz de probabilidades no formato [n_amostras, n_classes].
    """
    model_type = config.get("classificacao", {}).get("modelo_padrao", "xgboost")

    if model_type == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(models_dir / "catboost_model.cbm"))
        probs = np.asarray(model.predict_proba(X))
    elif model_type in {"sklearn_rf", "svm", "knn", "mlp"}:
        model_map = {
            "sklearn_rf": "rf_model.joblib",
            "svm": "svm_model.joblib",
            "knn": "knn_model.joblib",
            "mlp": "mlp_model.joblib",
        }
        model = joblib.load(models_dir / model_map[model_type])
        probs = np.asarray(model.predict_proba(X))
    elif model_type == "mlp_torch":
        artefato = carregar_checkpoint_mlp_torch(
            model_path=models_dir / "mlp_torch_model.pt",
            scaler_path=models_dir / "mlp_torch_scaler.joblib",
            device_cfg=config.get("classificacao", {}).get("mlp_torch", {}).get("device", "cuda"),
            logger=logger,
        )
        probs = predict_proba_mlp_torch(
            X=X,
            artefato=artefato,
            batch_size=int(config.get("classificacao", {}).get("mlp_torch", {}).get("batch_size", 1024)),
        )
    elif model_type == "siamese_torch":
        artefato = carregar_checkpoint_siamese_torch(
            model_path=models_dir / "siamese_torch_model.pt",
            scaler_path=models_dir / "siamese_torch_scaler.joblib",
            device_cfg=config.get("classificacao", {}).get("siamese_torch", {}).get("device", "cuda"),
            logger=logger,
        )
        probs = predict_proba_siamese_torch(
            X=X,
            artefato=artefato,
            batch_size=int(config.get("classificacao", {}).get("siamese_torch", {}).get("batch_size", 1024)),
        )
    else:
        model = xgb.XGBClassifier()
        model.load_model(str(models_dir / "xgboost_model.json"))
        probs = np.asarray(model.predict_proba(X))

    if probs.ndim == 1:
        probs = np.vstack([1.0 - probs, probs]).T
    return probs


def _resolver_imagem_classificacao(
    raw_cls_dir: Path,
    classe: str,
    arquivo: str,
    cache: Dict[str, Optional[Path]],
) -> Optional[Path]:
    """
    Resolve o caminho fisico da imagem de classificacao.

    Parametros:
        raw_cls_dir (Path): Diretorio raiz do dataset de classificacao.
        classe (str): Nome da classe esperada (pasta).
        arquivo (str): Nome do arquivo da imagem.
        cache (Dict[str, Optional[Path]]): Cache de resolucoes por arquivo.

    Retorno:
        Optional[Path]: Caminho resolvido, ou None se nao encontrado.
    """
    if arquivo in cache:
        return cache[arquivo]

    path_direto = raw_cls_dir / classe / arquivo
    if path_direto.exists():
        cache[arquivo] = path_direto
        return path_direto

    for p in raw_cls_dir.rglob(arquivo):
        if p.is_file():
            cache[arquivo] = p
            return p

    cache[arquivo] = None
    return None


def _reset_dataset(dataset_name: str) -> None:
    """
    Remove dataset existente do FiftyOne antes de recriar.

    Parametros:
        dataset_name (str): Nome do dataset a limpar.

    Retorno:
        None: Remove dataset se existir.
    """
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
