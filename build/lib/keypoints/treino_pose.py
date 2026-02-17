import logging
from pathlib import Path
import shutil
from typing import Dict, Any, List
from sklearn.model_selection import KFold
import yaml
from ultralytics import YOLO

from ..util.io_arquivos import garantir_diretorio, ler_yaml

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
    device = pose_cfg.get("device", "0")
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
        
        return _executar_yolo(model_name, yaml_path, epochs, imgsz, batch, device, runs_dir / "single", logger)

    else:
        # K-Fold Cross Validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_model_path = None
        
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
            final_model = _executar_yolo(model_name, yaml_fold, epochs, imgsz, batch, device, project_dir, logger)
            best_model_path = final_model # Guarda o último (ou implementar lógica para escolher o melhor mAP)
            
        logger.info("K-Fold concluído.")
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
    
    with open(train_txt, 'w') as f:
        for p in train_files:
            f.write(str(p.resolve()) + '\n')
            
    with open(val_txt, 'w') as f:
        for p in val_files:
            f.write(str(p.resolve()) + '\n')
            
    # Criar o YAML
    yaml_content = {
        "path": str(root_dir.resolve()), # Root dir (opcional se train/val forem absolutos)
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "kpt_shape": [8, 3],
        "names": {0: "cow"}
    }
    
    yaml_path = txt_dir / f"{name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    return yaml_path

def _executar_yolo(model_name: str, yaml_path: Path, epochs: int, imgsz: int, batch: int, device: str, project_dir: Path, logger: logging.Logger) -> Path:
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
