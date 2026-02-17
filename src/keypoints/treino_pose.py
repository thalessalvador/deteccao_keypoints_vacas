import logging
import torch
from pathlib import Path
import shutil
from typing import Dict, Any, List
from sklearn.model_selection import KFold
import yaml
from ultralytics import YOLO

from ..util.io_arquivos import garantir_diretorio, ler_yaml
from ..util.contratos import LISTA_KEYPOINTS_ORDENADA

def _ler_metricas_csv(csv_path: Path, fold_idx: int) -> Dict[str, Any]:
    """
    _ler_metricas_csv: Lê o arquivo results.csv gerado pelo YOLO e retorna as métricas da última época.

    Args:
        csv_path (Path): Caminho do arquivo results.csv gerado pelo YOLO.
        fold_idx (int): Índice do fold (1-based) para identificação na tabela.

    Returns:
        Dict[str, Any]: Dicionário com métricas (Box_mAP50, Box_mAP50-95, Pose_mAP50, Pose_mAP50-95).
        Retorna dicionário vazio se falhar a leitura ou arquivo não existir.
    """
    if not csv_path.exists():
        return {}
    
    import csv
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Pular cabeçalho
            # header geralmente tem nomes com espaços, ex: " metrics/mAP50(B)"
            header = [h.strip() for h in header]
            
            last_row = None
            for row in reader:
                last_row = row
                
            if not last_row:
                return {}
            
            def get_val(name):
                if name in header:
                    idx = header.index(name)
                    return float(last_row[idx].strip())
                return 0.0

            epoch = int(last_row[0].strip())
            
            return {
                "Fold": fold_idx,
                "Epochs": epoch,
                "Box_mAP50": get_val("metrics/mAP50(B)"),
                "Box_mAP50-95": get_val("metrics/mAP50-95(B)"),
                "Pose_mAP50": get_val("metrics/mAP50(P)"),
                "Pose_mAP50-95": get_val("metrics/mAP50-95(P)"),
            }
            
    except Exception as e:
        print(f"Erro ao ler CSV {csv_path}: {e}")
        return {}

def _imprimir_tabela_resumo(metrics_list: List[Dict[str, Any]], logger: logging.Logger):
    """
    _imprimir_tabela_resumo: Imprime uma tabela formatada com os resultados dos folds.

    Args:
        metrics_list (List[Dict[str, Any]]): Lista de dicionários com métricas de cada fold.
        logger (logging.Logger): Logger para imprimir a tabela.

    Returns:
        None
    """
    if not metrics_list:
        return

    # Cabeçalho
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
    
    # Média
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

    Suporta validação cruzada (K-Fold) se configurado.
    Se k_folds > 1, divide o dataset e treina K vezes, salvando os resultados.
    Retorna o caminho do 'melhor' modelo (do último fold ou lógica de seleção).

    Args:
        config (Dict[str, Any]): Configurações do sistema.
        dir_yolo (Path): Diretório do dataset YOLO (com images/ e labels/).
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho para o pesos do modelo treinado (best.pt).
    """
    pose_cfg = config.get("pose", {})
    k_folds = pose_cfg.get("k_folds", 1)
    imgsz = pose_cfg.get("imgsz", 640)
    batch = pose_cfg.get("batch", 16)
    epochs = pose_cfg.get("epochs", 100)
    patience = pose_cfg.get("patience", 50)
    
    device = pose_cfg.get("device", "0")
    if device != "cpu":
        if not torch.cuda.is_available():
            logger.warning(f"CUDA não disponível, mas device='{device}' foi solicitado. Forçando device='cpu'.")
            device = "cpu"
            
    model_name = pose_cfg.get("model_name", "yolov8n-pose.pt")
    
    # Listar todas as imagens
    images_dir = dir_yolo / "images"
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if not image_files:
        logger.error(f"Nenhuma imagem encontrada em {images_dir}")
        raise FileNotFoundError("Dataset vazio")

    logger.info(f"Iniciando treino. Dataset: {len(image_files)} imagens. K-Folds: {k_folds}")
    
    # Preparar diretório de runs
    runs_dir = Path("modelos/pose/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Se K=1, treino simples (sem split complexo, usa tudo ou split automatico do YOLO se definido val)
    # Mas como conversor jogou tudo em images/, precisamos definir validação.
    # Vamos assumir que se k=1, usamos 20% para validação se não houver pasta val separada.
    # O conversor do update anterior copia tudo para images/.
    
    if k_folds <= 1:
        # Modo simples: Treinar com split aleatório (YOLO faz se dermos fraction? Não, YOLO precisa de dataset.yaml com paths)
        # Vamos criar um split 80/20 manual
        train_files, val_files = _split_manual(image_files, 0.2)
        yaml_path = _criar_yaml_split(dir_yolo, train_files, val_files, "split_single", logger)
        
        return _executar_yolo(model_name, yaml_path, epochs, imgsz, batch, device, runs_dir / "single", logger, patience)

    else:
        # K-Fold Cross Validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_model_path = None
        
        fold_metrics = []

        for i, (train_index, val_index) in enumerate(kf.split(image_files)):
            fold_idx = i + 1
            logger.info(f"=== Iniciando Fold {fold_idx}/{k_folds} ===")
            
            fold_train_files = [image_files[j] for j in train_index]
            fold_val_files = [image_files[j] for j in val_index]
            
            logger.info(f"Fold {fold_idx}: {len(fold_train_files)} treino, {len(fold_val_files)} validação")
            
            # Criar YAML temporário para este fold
            yaml_fold = _criar_yaml_split(dir_yolo, fold_train_files, fold_val_files, f"split_fold_{fold_idx}", logger)
            
            project_dir = runs_dir / f"fold_{fold_idx}"
            
            # Treinar
            final_model = _executar_yolo(model_name, yaml_fold, epochs, imgsz, batch, device, project_dir, logger, patience)
            best_model_path = final_model # Guarda o último
            
            # Coletar métricas do results.csv
            metrics = _ler_metricas_csv(project_dir / "results.csv", fold_idx)
            if metrics:
                fold_metrics.append(metrics)
            
        logger.info("K-Fold concluído.")
        _imprimir_tabela_resumo(fold_metrics, logger)
        
        return best_model_path

def _split_manual(files: List[Path], val_frac: float) -> Any:
    """
    _split_manual: Realiza divisão aleatória de uma lista de arquivos.

    Args:
        files (List[Path]): Lista de caminhos de arquivos.
        val_frac (float): Fração de validação (ex: 0.2 para 20%).

    Returns:
        Tuple[List[Path], List[Path]]: Tupla (lista_treino, lista_validacao).
    """
    import random
    random.shuffle(files)
    n_val = int(len(files) * val_frac)
    return files[n_val:], files[:n_val]

def _criar_yaml_split(root_dir: Path, train_files: List[Path], val_files: List[Path], name: str, logger: logging.Logger) -> Path:
    """
    _criar_yaml_split: Gera um arquivo YAML de configuração de dataset para o YOLO.

    Cria arquivos .txt com a lista de caminhos absolutos das imagens de treino e validação,
    e então cria o dataset.yaml apontando para esses .txt.

    Args:
        root_dir (Path): Diretório raiz para salvar os splits.
        train_files (List[Path]): Arquivos de treino.
        val_files (List[Path]): Arquivos de validação.
        name (str): Identificador único para o split (ex: 'fold_1').
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho do arquivo YAML gerado.
    """
    # Criar arquivos .txt com listas de imagens
    txt_dir = root_dir / "splits"
    txt_dir.mkdir(exist_ok=True, parents=True)
    
    train_txt = txt_dir / f"{name}_train.txt"
    val_txt = txt_dir / f"{name}_val.txt"
    
    with open(train_txt, 'w', encoding='utf-8') as f:
        for p in train_files:
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

def _executar_yolo(model_name: str, yaml_path: Path, epochs: int, imgsz: int, batch: int, device: str, project_dir: Path, logger: logging.Logger, patience: int = 50) -> Path:
    """
    _executar_yolo: Instancia e inícia o processo de treinamento do YOLO.

    Args:
        model_name (str): Nome ou caminho do modelo base (ex: 'yolov8n-pose.pt').
        yaml_path (Path): Caminho do dataset.yaml configurado.
        epochs (int): Número de épocas.
        imgsz (int): Tamanho da imagem.
        batch (int): Tamanho do batch.
        device (str): Device ID (ex: '0' ou 'cpu').
        project_dir (Path): Diretório para salvar os resultados deste treino.
        logger (logging.Logger): Logger.

    Returns:
        Path: Caminho do arquivo de pesos do melhor modelo treinado (best.pt) ou last.pt.
    """
    try:
        model = YOLO(model_name)
        
        # Argumentos de augmentação podem ser passados via kwargs ou cfg file.
        # Aqui usamos os padrões ou configurados via arquivo se passarmos 'cfg'.
        # O Ultralytics usa configurações padrão se não sobrescritas. 
        # Para aplicar augs específico do config.yaml, precisariamos mapear para args do train.
        
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(project_dir.parent),
            name=project_dir.name,
            exist_ok=True,
            # Augmentations (mapear alguns principais)
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=2.0, # cuidado com shear
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.0, # Desligado conforme spec
            mosaic=0.7,
            mixup=0.1,
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


    
    import csv
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Pular cabeçalho
            # header geralmente tem nomes com espaços, ex: " metrics/mAP50(B)"
            header = [h.strip() for h in header]
            
            last_row = None
            for row in reader:
                last_row = row
                
            if not last_row:
                return {}
            
            # Mapear índices
            # Indices típicos no YOLOv8 Pose:
            # 0: epoch, ...
            # metrics/mAP50(B): index X
            # metrics/mAP50-95(B): index Y
            # metrics/mAP50(P): index Z
            # metrics/mAP50-95(P): index W
            
            def get_val(name):
                if name in header:
                    idx = header.index(name)
                    return float(last_row[idx].strip())
                return 0.0

            epoch = int(last_row[0].strip())
            
            return {
                "Fold": fold_idx,
                "Epochs": epoch,
                "Box_mAP50": get_val("metrics/mAP50(B)"),
                "Box_mAP50-95": get_val("metrics/mAP50-95(B)"),
                "Pose_mAP50": get_val("metrics/mAP50(P)"),
                "Pose_mAP50-95": get_val("metrics/mAP50-95(P)"),
            }
            
    except Exception as e:
        print(f"Erro ao ler CSV {csv_path}: {e}")
        return {}

def _imprimir_tabela_resumo(metrics_list: List[Dict[str, Any]], logger: logging.Logger):
    """Imprime uma tabela formatada com os resultados dos folds."""
    if not metrics_list:
        return

    # Cabeçalho
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
    
    # Média
    if count > 0:
        avg_box_50 = soma_box_50 / count
        avg_box_95 = soma_box_95 / count
        avg_pose_50 = soma_pose_50 / count
        avg_pose_95 = soma_pose_95 / count
        
        logger.info(f"{'MEDIA':^6} | {'-':^8} | {avg_box_50:^12.4f} | {avg_box_95:^14.4f} | {avg_pose_50:^12.4f} | {avg_pose_95:^15.4f}")
    
    logger.info("="*80 + "\n")
