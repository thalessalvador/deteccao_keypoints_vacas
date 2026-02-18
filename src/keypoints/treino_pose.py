import logging
import torch
from pathlib import Path
import shutil
from typing import Dict, Any, List
from collections import deque
from sklearn.model_selection import KFold, GroupKFold
import re
import yaml
from ultralytics import YOLO

from ..util.io_arquivos import garantir_diretorio, ler_yaml
from ..util.contratos import LISTA_KEYPOINTS_ORDENADA

def _extrair_grupo_validacao(caminho_imagem: Path, estrategia: str) -> str:
    """
    Extrai chave de grupo para GroupKFold a partir do nome da imagem.

    Estrategias:
    - groupkfold_por_sessao: data + baia + camera
    - groupkfold_por_anotador: proxy por prefixo + camera
    """
    stem = caminho_imagem.stem
    tokens = stem.split("_")
    estrategia = (estrategia or "").lower()

    # Padrao 1 (Cows Challenge):
    # cow_id_YYYY_MM_DD_HH_MM_SS_cam_ID_station_id
    p1 = re.match(
        r"^(?P<cow>[^_]+)_(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})_(?P<h>\d{2})_(?P<mi>\d{2})_(?P<s>\d{2})_cam_(?P<cam>[^_]+)_(?P<station>.+)$",
        stem,
    )

    # Padrao 2a (Cows Challenge):
    # cam_ID_XX_YYYYMMDDHHMMSS_station_id_cam_ID
    p2a = re.match(
        r"^cam_(?P<cam1>[^_]+)_[^_]+_(?P<dt>\d{14})_(?P<station>.+?)_cam_(?P<cam2>[^_]+)$",
        stem,
    )

    # Padrao 2b (Cows Challenge):
    # YYYYMMDD_HHMMSS_station_id_cam_ID
    p2b = re.match(
        r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<station>.+?)_cam_(?P<cam>[^_]+)$",
        stem,
    )

    # Fallback usado no dataset atual:
    # YYYYMMDD_HHMMSS_baiaNN_IPC1
    p_fb = re.match(
        r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<station>[^_]+)_(?P<cam>[^_]+)$",
        stem,
    )

    data = "data_desconhecida"
    station = "station_desconhecida"
    camera = tokens[-1] if tokens else stem
    prefixo = tokens[0] if tokens else stem

    if p1:
        data = f"{p1.group('y')}{p1.group('m')}{p1.group('d')}"
        station = p1.group("station")
        camera = p1.group("cam")
        prefixo = p1.group("cow")
    elif p2a:
        data = p2a.group("dt")[:8]
        station = p2a.group("station")
        camera = p2a.group("cam2")
        prefixo = f"cam_{p2a.group('cam1')}"
    elif p2b:
        data = p2b.group("date")
        station = p2b.group("station")
        camera = p2b.group("cam")
    elif p_fb:
        data = p_fb.group("date")
        station = p_fb.group("station")
        camera = p_fb.group("cam")
    else:
        baia = next((t for t in tokens if t.lower().startswith("baia")), None)
        if baia:
            station = baia
        m_data = re.search(r"(20\d{6})", stem)
        if m_data:
            data = m_data.group(1)

    if estrategia == "groupkfold_por_anotador":
        return f"{prefixo}_{camera}"

    return f"{data}_{station}_{camera}"

def _ler_metricas_csv(csv_path: Path, fold_idx: int) -> Dict[str, Any]:
    """
    _ler_metricas_csv: LÃª o arquivo results.csv gerado pelo YOLO e retorna as mÃ©tricas da Ãºltima Ã©poca.

    Args:
        csv_path (Path): Caminho do arquivo results.csv gerado pelo YOLO.
        fold_idx (int): Ãndice do fold (1-based) para identificaÃ§Ã£o na tabela.

    Returns:
        Dict[str, Any]: DicionÃ¡rio com mÃ©tricas (Box_mAP50, Box_mAP50-95, Pose_mAP50, Pose_mAP50-95).
        Retorna dicionÃ¡rio vazio se falhar a leitura ou arquivo nÃ£o existir.
    """
    if not csv_path.exists():
        return {}
    
    import csv
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Pular cabeÃ§alho
            # header geralmente tem nomes com espaÃ§os, ex: " metrics/mAP50(B)"
            header = [h.strip() for h in header]
            
            last_row = None
            for row in reader:
                if row: # evitar linhas vazias
                    last_row = row
                
            if not last_row:
                return {}
            
            def get_val(name: str) -> float:
                """Retorna valor float da metrica solicitada na ultima linha do CSV."""
                # Tenta encontrar correspondencia exata apÃ³s strip
                name_clean = name.strip()
                if name_clean in header:
                    idx = header.index(name_clean)
                    try:
                        return float(last_row[idx].strip())
                    except ValueError:
                        return 0.0
                return 0.0

            try:
                epoch = int(last_row[0].strip())
            except ValueError:
                epoch = 0
            
            return {
                "Fold": fold_idx,
                "Epochs": epoch,
                "Box_mAP50": get_val("metrics/mAP50(B)"),
                "Box_mAP50-95": get_val("metrics/mAP50-95(B)"),
                "Pose_mAP50": get_val("metrics/mAP50(P)"),
                "Pose_mAP50-95": get_val("metrics/mAP50-95(P)"),
            }
            
    except Exception as e:
        logging.getLogger("projeto_vacas").warning(f"Erro ao ler CSV {csv_path}: {e}")
        return {}

def _imprimir_tabela_resumo(metrics_list: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """
    _imprimir_tabela_resumo: Imprime uma tabela formatada com os resultados dos folds.

    Args:
        metrics_list (List[Dict[str, Any]]): Lista de dicionÃ¡rios com mÃ©tricas de cada fold.
        logger (logging.Logger): Logger para imprimir a tabela.

    Returns:
        None
    """
    if not metrics_list:
        return

    # CabeÃ§alho
    logger.info("\n" + "="*80)
    logger.info(f"{'Fold':^6} | {'Epocas':^8} | {'Box mAP50':^12} | {'Box mAP50-95':^14} | {'Pose mAP50':^12} | {'Pose mAP50-95':^15}")
    logger.info("-" * 80)
    
    # Linhas
    soma_box_50 = 0
    soma_box_95 = 0
    soma_pose_50 = 0
    soma_pose_95 = 0
    
    count = len(metrics_list)
    
    for m in metrics_list:
        logger.info(f"{m['Fold']:^6} | {m['Epochs']:^8} | {m['Box_mAP50']:^12.4f} | {m['Box_mAP50-95']:^14.4f} | {m['Pose_mAP50']:^12.4f} | {m['Pose_mAP50-95']:^15.4f}")
        
        soma_box_50 += m['Box_mAP50']
        soma_box_95 += m['Box_mAP50-95']
        soma_pose_50 += m['Pose_mAP50']
        soma_pose_95 += m['Pose_mAP50-95']
        
    logger.info("-" * 80)
    
    # MÃ©dia
    if count > 0:
        avg_box_50 = soma_box_50 / count
        avg_box_95 = soma_box_95 / count
        avg_pose_50 = soma_pose_50 / count
        avg_pose_95 = soma_pose_95 / count
        
        logger.info(f"{'MEDIA':^6} | {'-':^8} | {avg_box_50:^12.4f} | {avg_box_95:^14.4f} | {avg_pose_50:^12.4f} | {avg_pose_95:^15.4f}")
    
    logger.info("="*80 + "\n")

def treinar_modelo_pose(config: Dict[str, Any], dir_yolo: Path, logger: logging.Logger) -> Path:
    """
    treinar_modelo_pose: Executa o treinamento do modelo YOLOv8 Pose.

    Suporta validaÃ§Ã£o cruzada (K-Fold) se configurado.
    Se k_folds > 1, divide o dataset e treina K vezes, salvando os resultados.
    Retorna o caminho do 'melhor' modelo (do Ãºltimo fold ou lÃ³gica de seleÃ§Ã£o).

    Args:
        config (Dict[str, Any]): ConfiguraÃ§Ãµes do sistema.
        dir_yolo (Path): DiretÃ³rio do dataset YOLO (com images/ e labels/).
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho para o pesos do modelo treinado (best.pt).
    """
    pose_cfg = config.get("pose", {})
    k_folds = pose_cfg.get("k_folds", 1)
    estrategia_validacao = pose_cfg.get("estrategia_validacao", "kfold_misturado")
    forcar_diversidade_batch = bool(pose_cfg.get("forcar_diversidade_batch", False))
    imgsz = pose_cfg.get("imgsz", 640)
    batch = pose_cfg.get("batch", 16)
    workers = int(pose_cfg.get("workers", 8))
    cache = pose_cfg.get("cache", False)
    amp = bool(pose_cfg.get("amp", True))
    epochs = pose_cfg.get("epochs", 100)
    patience = pose_cfg.get("patience", 50)
    
    device = pose_cfg.get("device", "0")
    if device != "cpu":
        if not torch.cuda.is_available():
            logger.warning(f"CUDA nÃ£o disponÃ­vel, mas device='{device}' foi solicitado. ForÃ§ando device='cpu'.")
            device = "cpu"
            
    model_name = pose_cfg.get("model_name", "yolov8n-pose.pt")
    usar_aug = bool(pose_cfg.get("usar_data_augmentation", True))
    aug_cfg = pose_cfg.get("augmentacao", {}) if usar_aug else {}
    
    # Listar todas as imagens
    images_dir = dir_yolo / "images"
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if not image_files:
        logger.error(f"Nenhuma imagem encontrada em {images_dir}")
        raise FileNotFoundError("Dataset vazio")

    logger.info(
        f"Iniciando treino. Dataset: {len(image_files)} imagens. "
        f"K-Folds: {k_folds}. Estrategia validacao: {estrategia_validacao}"
    )
    
    # Preparar diretÃ³rio de runs
    runs_dir = Path("modelos/pose/runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Se K=1, treino simples (sem split complexo, usa tudo ou split automatico do YOLO se definido val)
    # Mas como conversor jogou tudo em images/, precisamos definir validaÃ§Ã£o.
    # Vamos assumir que se k=1, usamos 20% para validaÃ§Ã£o se nÃ£o houver pasta val separada.
    # O conversor do update anterior copia tudo para images/.
    
    if k_folds <= 1:
        # Modo simples: Treinar com split aleatÃ³rio (YOLO faz se dermos fraction? NÃ£o, YOLO precisa de dataset.yaml com paths)
        # Vamos criar um split 80/20 manual
        train_files, val_files = _split_manual(image_files, 0.2)
        yaml_path = _criar_yaml_split(
            dir_yolo,
            train_files,
            val_files,
            "split_single",
            logger,
            forcar_diversidade_batch=forcar_diversidade_batch,
            estrategia_grupo_diversidade=estrategia_validacao,
        )
        
        return _executar_yolo(
            model_name,
            yaml_path,
            epochs,
            imgsz,
            batch,
            device,
            runs_dir / "single",
            logger,
            patience,
            usar_aug,
            aug_cfg,
            workers,
            cache,
            amp,
        )

    else:
        # K-Fold Cross Validation
        estrategia_norm = str(estrategia_validacao).lower()
        usar_groupkfold = estrategia_norm.startswith("groupkfold")
        split_iter = None
        grupos = None
        if usar_groupkfold:
            grupos = [_extrair_grupo_validacao(p, estrategia_norm) for p in image_files]
            n_grupos = len(set(grupos))
            logger.info(f"GroupKFold habilitado. Grupos unicos detectados: {n_grupos}")
            if n_grupos >= k_folds:
                gkf = GroupKFold(n_splits=k_folds)
                split_iter = gkf.split(image_files, groups=grupos)
            else:
                logger.warning(
                    f"GroupKFold solicitado, mas ha apenas {n_grupos} grupos para {k_folds} folds. "
                    "Fallback para KFold misturado."
                )
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                split_iter = kf.split(image_files)
        else:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            split_iter = kf.split(image_files)

        best_model_path = None
        best_map_pose = -1.0
        
        fold_metrics = []

        for i, (train_index, val_index) in enumerate(split_iter):
            fold_idx = i + 1
            logger.info(f"=== Iniciando Fold {fold_idx}/{k_folds} ===")
            
            fold_train_files = [image_files[j] for j in train_index]
            fold_val_files = [image_files[j] for j in val_index]
            
            logger.info(f"Fold {fold_idx}: {len(fold_train_files)} treino, {len(fold_val_files)} validacao")
            
            # Criar YAML temporÃ¡rio para este fold
            yaml_fold = _criar_yaml_split(
                dir_yolo,
                fold_train_files,
                fold_val_files,
                f"split_fold_{fold_idx}",
                logger,
                forcar_diversidade_batch=forcar_diversidade_batch,
                estrategia_grupo_diversidade=estrategia_validacao,
            )
            
            project_dir = runs_dir / f"fold_{fold_idx}"
            
            # Treinar
            final_model = _executar_yolo(
                model_name,
                yaml_fold,
                epochs,
                imgsz,
                batch,
                device,
                project_dir,
                logger,
                patience,
                usar_aug,
                aug_cfg,
                workers,
                cache,
                amp,
            )
            
            # Coletar mÃ©tricas do results.csv
            metrics = _ler_metricas_csv(project_dir / "results.csv", fold_idx)
            if metrics:
                fold_metrics.append(metrics)
                
                # Verifica se Ã© o melhor modelo atÃ© agora (baseado em Pose mAP50-95)
                # Se nÃ£o tiver Pose metrics (ex: dataset sÃ³ bbox), usa Box mAP50-95
                current_map = metrics.get("Pose_mAP50-95", 0.0)
                if current_map == 0.0:
                     current_map = metrics.get("Box_mAP50-95", 0.0)
                     
                logger.info(f"Fold {fold_idx} mAP50-95: {current_map:.4f}")
                
                if current_map > best_map_pose:
                    best_map_pose = current_map
                    best_model_path = final_model
                    logger.info(f"Novo melhor modelo detectado: Fold {fold_idx}")
            else:
                 # Fallback se nÃ£o conseguir ler mÃ©tricas: assume o Ãºltimo como best se ainda n tinha
                 if best_model_path is None:
                     best_model_path = final_model
            
        logger.info("K-Fold concluído.")
        _imprimir_tabela_resumo(fold_metrics, logger)
        
        # Salvar relatÃ³rio JSON
        relatorio = {
            "folds": fold_metrics,
            "melhor_modelo": {
                "path": str(best_model_path) if best_model_path else None,
                "map50_95": best_map_pose
            }
        }
        
        relatorio_path = Path("saidas/relatorios/metricas_pose.json").resolve()
        relatorio_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(relatorio_path, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, indent=2)
            
        logger.info(f"Relatório de métricas salvo em: {relatorio_path}")
        
        if best_model_path:
            logger.info(f"Selecionado o modelo do Fold com melhor mAP ({best_map_pose:.4f}): {best_model_path}")
        
        return best_model_path



def _split_manual(files: List[Path], val_frac: float) -> Any:
    """
    _split_manual: Realiza divisÃ£o aleatÃ³ria de uma lista de arquivos.

    Args:
        files (List[Path]): Lista de caminhos de arquivos.
        val_frac (float): FraÃ§Ã£o de validaÃ§Ã£o (ex: 0.2 para 20%).

    Returns:
        Tuple[List[Path], List[Path]]: Tupla (lista_treino, lista_validacao).
    """
    import random
    random.shuffle(files)
    n_val = int(len(files) * val_frac)
    return files[n_val:], files[:n_val]

def _intercalar_treino_por_grupo(
    arquivos_treino: List[Path],
    estrategia_grupo: str,
) -> List[Path]:
    """
    _intercalar_treino_por_grupo: Reordena treino para alternar grupos e aumentar diversidade local.

    A reordenacao usa round-robin entre grupos extraidos do nome do arquivo.
    Nao altera o conjunto de amostras; apenas a ordem de escrita no TXT de treino.

    Args:
        arquivos_treino (List[Path]): Arquivos do split de treino.
        estrategia_grupo (str): Estrategia usada para extracao do grupo.

    Returns:
        List[Path]: Lista reordenada, priorizando alternancia entre grupos.
    """
    grupos_para_arquivos: Dict[str, List[Path]] = {}
    for caminho in arquivos_treino:
        grupo = _extrair_grupo_validacao(caminho, estrategia_grupo)
        grupos_para_arquivos.setdefault(grupo, []).append(caminho)

    for grupo, arquivos in grupos_para_arquivos.items():
        grupos_para_arquivos[grupo] = sorted(arquivos)

    fila_grupos = deque(sorted(grupos_para_arquivos.keys()))
    arquivos_ordenados: List[Path] = []

    while fila_grupos:
        grupo_atual = fila_grupos.popleft()
        arquivos_grupo = grupos_para_arquivos[grupo_atual]
        arquivos_ordenados.append(arquivos_grupo.pop(0))
        if arquivos_grupo:
            fila_grupos.append(grupo_atual)

    return arquivos_ordenados

def _criar_yaml_split(
    root_dir: Path,
    train_files: List[Path],
    val_files: List[Path],
    name: str,
    logger: logging.Logger,
    forcar_diversidade_batch: bool = False,
    estrategia_grupo_diversidade: str = "groupkfold_por_sessao",
) -> Path:
    """
    _criar_yaml_split: Gera um arquivo YAML de configuraÃ§Ã£o de dataset para o YOLO.

    Cria arquivos .txt com a lista de caminhos absolutos das imagens de treino e validaÃ§Ã£o,
    e entÃ£o cria o dataset.yaml apontando para esses .txt.

    Args:
        root_dir (Path): DiretÃ³rio raiz para salvar os splits.
        train_files (List[Path]): Arquivos de treino.
        val_files (List[Path]): Arquivos de validaÃ§Ã£o.
        name (str): Identificador Ãºnico para o split (ex: 'fold_1').
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho do arquivo YAML gerado.
    """
    # Criar arquivos .txt com listas de imagens
    txt_dir = root_dir / "splits"
    txt_dir.mkdir(exist_ok=True, parents=True)
    
    train_txt = txt_dir / f"{name}_train.txt"
    val_txt = txt_dir / f"{name}_val.txt"
    
    arquivos_treino_saida = train_files
    if forcar_diversidade_batch:
        arquivos_treino_saida = _intercalar_treino_por_grupo(
            arquivos_treino=train_files,
            estrategia_grupo=estrategia_grupo_diversidade,
        )
        grupos_unicos = len({
            _extrair_grupo_validacao(p, estrategia_grupo_diversidade)
            for p in arquivos_treino_saida
        })
        logger.info(
            f"Diversidade de batch habilitada no split '{name}': "
            f"treino reordenado por grupos ({grupos_unicos} grupos)."
        )

    with open(train_txt, 'w', encoding='utf-8') as f:
        for p in arquivos_treino_saida:
            f.write(str(p.resolve()) + '\n')
            
    with open(val_txt, 'w', encoding='utf-8') as f:
        for p in val_files:
            f.write(str(p.resolve()) + '\n')
            
    # Criar o YAML
    yaml_content = {
        "path": str(root_dir.resolve()), # Root dir (opcional se train/val forem absolutos)
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "kpt_shape": [8, 3],
        "names": {0: "cow"},
        "kpt_names": LISTA_KEYPOINTS_ORDENADA
    }
    
    yaml_path = txt_dir / f"{name}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    return yaml_path

def _executar_yolo(
    model_name: str,
    yaml_path: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project_dir: Path,
    logger: logging.Logger,
    patience: int = 50,
    usar_aug: bool = True,
    aug_cfg: Dict[str, Any] = None,
    workers: int = 8,
    cache: Any = False,
    amp: bool = True,
) -> Path:
    """
    _executar_yolo: Instancia e inÃ­cia o processo de treinamento do YOLO.

    Args:
        model_name (str): Nome ou caminho do modelo base (ex: 'yolov8n-pose.pt').
        yaml_path (Path): Caminho do dataset.yaml configurado.
        epochs (int): NÃºmero de Ã©pocas.
        imgsz (int): Tamanho da imagem.
        batch (int): Tamanho do batch.
        device (str): Device ID (ex: '0' ou 'cpu').
        project_dir (Path): DiretÃ³rio para salvar os resultados deste treino.
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho do arquivo de pesos do melhor modelo treinado (best.pt) ou last.pt.
    """
    try:
        model = YOLO(model_name)

        aug_cfg = aug_cfg or {}
        if usar_aug:
            logger.info("Pose augmentation habilitada (valores do config).")
            train_aug_kwargs = {
                "hsv_h": float(aug_cfg.get("hsv_h", 0.015)),
                "hsv_s": float(aug_cfg.get("hsv_s", 0.7)),
                "hsv_v": float(aug_cfg.get("hsv_v", 0.4)),
                "degrees": float(aug_cfg.get("degrees", 5.0)),
                "translate": float(aug_cfg.get("translate", 0.1)),
                "scale": float(aug_cfg.get("scale", 0.5)),
                "shear": float(aug_cfg.get("shear", 2.0)),
                "perspective": float(aug_cfg.get("perspective", 0.0005)),
                "flipud": float(aug_cfg.get("flipud", 0.0)),
                "fliplr": float(aug_cfg.get("fliplr", 0.0)),
                "mosaic": float(aug_cfg.get("mosaic", 0.7)),
                "mixup": float(aug_cfg.get("mixup", 0.1)),
                "erasing": float(aug_cfg.get("erasing", 0.2)),
            }
        else:
            logger.info("Pose augmentation desabilitada (usar_data_augmentation=false).")
            train_aug_kwargs = {
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "erasing": 0.0,
            }
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            cache=cache,
            amp=amp,
            device=device,
            project=str(project_dir.parent),
            name=project_dir.name,
            exist_ok=True,
            **train_aug_kwargs,
            patience=patience,
        )
        
        # Retorna caminho do best.pt
        best_pt = project_dir / "weights" / "best.pt"
        if best_pt.exists():
            logger.info(f"Modelo salvo em {best_pt}")
            return best_pt
        return project_dir / "weights" / "last.pt"
        
    except Exception as e:
        logger.error(f"Erro no treino YOLO: {e}")
        raise e



