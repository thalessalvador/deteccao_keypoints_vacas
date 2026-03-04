import argparse
import sys
from pathlib import Path
import logging
import shutil
from typing import List

from .util.logging_cfg import configurar_logger
from .util.io_arquivos import ler_yaml, garantir_diretorio
from .keypoints.parser_label_studio import carregar_anotacoes_label_studio
from .keypoints.conversor_yolo_pose import converter_para_yolo_pose, gerar_dataset_yaml_ultralytics
from .keypoints.treino_pose import treinar_modelo_pose
from .keypoints.inferencia_pose import inferir_keypoints_em_imagem
from .classificacao.gerador_dataset_features import gerar_dataset_features
from .classificacao.treino_classificador import treinar_classificador
from .classificacao.avaliacao_classificador import avaliar_classificador
from .analise_features.analise_exploratoria import analisar_features_exploratorio


def _remover_arquivo_ou_pasta(caminho: Path, logger: logging.Logger) -> None:
    """
    Remove um arquivo ou diretorio se existir.
    
    Args:
        caminho (Path): Caminho a remover.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        None: Executa remocao quando possivel e registra o resultado no log.
    """
    try:
        if not caminho.exists():
            return
        if caminho.is_dir():
            shutil.rmtree(caminho)
            logger.info(f"Limpeza: diretório removido -> {caminho}")
        else:
            caminho.unlink()
            logger.info(f"Limpeza: arquivo removido -> {caminho}")
    except Exception as exc:
        logger.warning(f"Nao foi possivel remover '{caminho}': {exc}")


def _remover_por_padroes(base_dir: Path, padroes: List[str], logger: logging.Logger) -> None:
    """
    Remove arquivos em um diretorio de acordo com padroes glob.
    
    Args:
        base_dir (Path): Diretorio base para aplicar os padroes.
        padroes (List[str]): Padroes glob de arquivos.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        None: Remove os arquivos encontrados e registra as acoes no log.
    """
    if not base_dir.exists():
        return

    for padrao in padroes:
        for arquivo in base_dir.glob(padrao):
            _remover_arquivo_ou_pasta(arquivo, logger)


def _limpar_resultados_subcomando(config: dict, subcomando: str, logger: logging.Logger) -> None:
    """
    Limpa artefatos da execucao anterior de um subcomando.
    
    Args:
        config (dict): Configuracoes carregadas.
        subcomando (str): Nome do subcomando da CLI.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        None: Remove arquivos/diretorios de saida relacionados ao subcomando.
    """
    processed_dir = Path(config["paths"]["processed"])
    models_dir = Path(config["paths"]["models"])
    outputs_dir = Path(config["paths"]["outputs"])
    relatorios_dir = outputs_dir / "relatorios"

    if subcomando == "preprocessar-pose":
        _remover_arquivo_ou_pasta(processed_dir / "yolo_pose", logger)
        return

    if subcomando == "treinar-pose":
        _remover_arquivo_ou_pasta(models_dir / "pose" / "runs", logger)
        _remover_arquivo_ou_pasta(relatorios_dir / "metricas_pose.json", logger)
        return

    if subcomando == "gerar-features":
        _remover_arquivo_ou_pasta(processed_dir / "classificacao" / "features", logger)
        _remover_arquivo_ou_pasta(processed_dir / "classificacao" / "splits", logger)
        return

    if subcomando == "treinar-classificador":
        _remover_arquivo_ou_pasta(models_dir / "classificacao" / "modelos_salvos", logger)
        _remover_por_padroes(
            relatorios_dir,
            [
                "metricas_classificacao_treino.json",
                "xgb_*.png",
                "rf_*.png",
                "catboost_*.png",
                "svm_*.png",
                "mlp_*.png",
                "siamese_torch_*.png",
            ],
            logger,
        )
        return

    if subcomando == "avaliar-classificador":
        _remover_por_padroes(
            relatorios_dir,
            [
                "matriz_confusao.csv",
                "matriz_confusao.png",
                "metricas_por_classe.png",
                "confianca_corretas_vs_incorretas.png",
                "cobertura_vs_acuracia.png",
                "metricas_classificacao.json",
            ],
            logger,
        )
        return

def _limpar_resultados_pipeline_completo(config: dict, logger: logging.Logger) -> None:
    """
    Limpa resultados anteriores das etapas do pipeline completo.
    
    Args:
        config (dict): Configuracoes carregadas.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        None: Executa limpeza sequencial das etapas previstas no pipeline completo.
    """
    etapas = [
        "preprocessar-pose",
        "treinar-pose",
        "gerar-features",
        "treinar-classificador",
        "avaliar-classificador",
    ]
    for etapa in etapas:
        _limpar_resultados_subcomando(config, etapa, logger)

def main() -> None:
    """
    Ponto de entrada da CLI do projeto.
    
    Monta argumentos, carrega configuracao e direciona a execucao para o subcomando solicitado.
    
    Args:
        Nenhum.
    
    Returns:
        None: Executa o fluxo correspondente ao comando informado.
    """
    parser = argparse.ArgumentParser(description="CLI Projeto Vacas")
    subparsers = parser.add_subparsers(dest="comando", help="Subcomandos disponíveis")
    
    # Argumento global config
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Caminho do arquivo de configuração")

    # 1. Preprocessar Pose
    cmd_prep = subparsers.add_parser("preprocessar-pose", help="Converte Label Studio -> YOLO Pose")
    
    # 2. Treinar Pose
    cmd_train = subparsers.add_parser("treinar-pose", help="Treina modelo YOLO Pose")
    
    # 4. Inferir Pose
    cmd_inf = subparsers.add_parser("inferir-pose", help="Roda inferência em imagem")
    cmd_inf.add_argument("--imagem", type=str, required=True, help="Imagem para inferência")
    cmd_inf.add_argument("--desenhar", action="store_true", help="Salvar imagem com plot")
    cmd_inf.add_argument(
        "--desenhar-augmentacao",
        action="store_true",
        help="Salvar preview com keypoints originais + copias com ruido gaussiano"
    )
    cmd_inf.add_argument(
        "--aug-copias",
        type=int,
        default=None,
        help="Numero de copias no preview de augmentacao (padrao: config)"
    )
    
    # 5. Gerar Features
    cmd_feats = subparsers.add_parser("gerar-features", help="Gera features geométricas para classificação")

    # 6. Treinar Classificador
    cmd_train_cls = subparsers.add_parser("treinar-classificador", help="Treina classificador (XGBoost)")

    # 7. Avaliar Classificador
    cmd_eval_cls = subparsers.add_parser("avaliar-classificador", help="Avalia classificador (Matriz Confusão)")

    # 8. Classificar Imagem Única
    cmd_cls_img = subparsers.add_parser("classificar-imagem", help="Classifica uma imagem única (End-to-End)")
    cmd_cls_img.add_argument("--imagem", type=str, required=True, help="Caminho da imagem")
    cmd_cls_img.add_argument("--top-k", type=int, default=3, help="Número de predições")
    cmd_cls_img.add_argument("--desenhar", action="store_true", help="Salvar imagem com predição")

    # 9. Pipeline Completo
    cmd_pipe = subparsers.add_parser("pipeline-completo", help="Roda pipeline completo")
    
    # 10. Analise de Features (EDA)
    cmd_eda = subparsers.add_parser("analisar-features", help="Roda EDA das features de classificacao")

    args = parser.parse_args()
    
    # Setup Logger e Config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Erro: Arquivo de config {config_path} não encontrado.")
        sys.exit(1)
        
    config = ler_yaml(config_path)
    logger = configurar_logger()
    
    if args.comando == "preprocessar-pose":
        _limpar_resultados_subcomando(config, "preprocessar-pose", logger)
        run_preprocessar_pose(config, logger)
        
    elif args.comando == "treinar-pose":
        _limpar_resultados_subcomando(config, "treinar-pose", logger)
        run_treinar_pose(config, logger)
        
    elif args.comando == "inferir-pose":
        run_inferir_pose(
            config,
            args.imagem,
            args.desenhar,
            logger,
            desenhar_augmentacao=args.desenhar_augmentacao,
            aug_copias=args.aug_copias,
        )
    
    elif args.comando == "gerar-features":
        _limpar_resultados_subcomando(config, "gerar-features", logger)
        run_gerar_features(config, logger)
        
    elif args.comando == "treinar-classificador":
        _limpar_resultados_subcomando(config, "treinar-classificador", logger)
        run_treinar_classificador(config, logger)

    elif args.comando == "avaliar-classificador":
        _limpar_resultados_subcomando(config, "avaliar-classificador", logger)
        run_avaliar_classificador(config, logger)

    elif args.comando == "classificar-imagem":
        run_classificar_imagem(config, args.imagem, args.top_k, args.desenhar, logger)

    elif args.comando == "pipeline-completo":
        logger.info("Iniciando Pipeline Completo...")
        _limpar_resultados_pipeline_completo(config, logger)
        run_preprocessar_pose(config, logger)
        model_path = run_treinar_pose(config, logger)
        
        # Fase 2
        run_gerar_features(config, logger)
        
        # Fase 3
        run_treinar_classificador(config, logger)
        run_avaliar_classificador(config, logger)
    
    elif args.comando == "analisar-features":
        run_analisar_features(config, logger)
        
    else:
        parser.print_help()

def run_treinar_classificador(config: dict, logger: logging.Logger) -> Path:
    """
    run_treinar_classificador: Executa a etapa de treino do classificador.
    
    Args:
        config (dict): Dicionario de configuracao.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        Path: Caminho do modelo treinado salvo em disco.
    """
    return treinar_classificador(config, logger)

def run_avaliar_classificador(config: dict, logger: logging.Logger) -> None:
    """
    run_avaliar_classificador: Executa a avaliacao do classificador.
    
    Args:
        config (dict): Dicionario de configuracao.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        None: Gera metricas e relatorios de avaliacao no diretorio de saida.
    """
    avaliar_classificador(config, logger)


def run_preprocessar_pose(config: dict, logger: logging.Logger) -> None:
    """
    run_preprocessar_pose: Executa a etapa de pre-processamento (Fase 1).
    
    Carrega anotacoes do Label Studio, converte para YOLO Pose e gera dataset.yaml.
    
    Args:
        config (dict): Dicionario de configuracao.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        None: Persiste dataset processado e arquivos auxiliares de configuracao.
    """
    logger.info("=== Fase 1: Pré-processamento (Label Studio -> YOLO) ===")
    raw_path = Path(config["paths"]["raw"]) / "dataset_keypoints"
    processed_path = Path(config["paths"]["processed"]) / "yolo_pose"
    
    # 1. Carregar
    imagens_anotadas = carregar_anotacoes_label_studio(raw_path, logger)
    if not imagens_anotadas:
        logger.error("Nenhuma anotação carregada. Abortando.")
        return

    # 2. Converter
    converter_para_yolo_pose(imagens_anotadas, processed_path, logger)
    
    # 3. Gerar YAML Base
    gerar_dataset_yaml_ultralytics(processed_path, logger)
    
def run_treinar_pose(config: dict, logger: logging.Logger) -> Path:
    """
    run_treinar_pose: Executa a etapa de treinamento do modelo de pose.

    Args:
        config (dict): Dicionário de configuração.
        logger (logging.Logger): Logger configurado.

    Returns:
        Path: Caminho do melhor modelo treinado.
    """
    logger.info("=== Fase 1: Treino YOLO Pose ===")
    processed_path = Path(config["paths"]["processed"]) / "yolo_pose"
    
    best_model = treinar_modelo_pose(config, processed_path, logger)
    logger.info(f"Treino finalizado. Melhor modelo: {best_model}")
    return best_model

def run_inferir_pose(
    config: dict,
    img_path_str: str,
    desenhar: bool,
    logger: logging.Logger,
    desenhar_augmentacao: bool = False,
    aug_copias: int | None = None,
) -> None:
    """
    run_inferir_pose: Executa inferencia de pose em uma imagem unica.
    
    Args:
        config (dict): Dicionario de configuracao.
        img_path_str (str): Caminho da imagem.
        desenhar (bool): Flag para gerar imagem com plot.
        logger (logging.Logger): Logger configurado.
        desenhar_augmentacao (bool): Flag para gerar preview de augmentacao gaussiana.
        aug_copias (int | None): Numero de copias para preview de augmentacao.
    
    Returns:
        None: Imprime JSON de inferencia e salva imagens de saida quando solicitado.
    """
    logger.info(f"=== Fase 1: Inferência em {img_path_str} ===")
    img_path = Path(img_path_str)
    import json
    
    # Pegar modelo do config ou usar padrão treinado/runs (aqui assumindo um path fixo ou placeholder)
    # Tentar pegar do output de treino se existir, ou do config 'model_name' inicial se não tiver treinado ainda (vai baixar)
    # Mas idealmente queremos o treinado.
    # Vamos procurar em modelos/pose/runs/fold_X/weights/best.pt ou similar.
    # Por simplicidade, vamos tentar o path fixo de um treino 'single' ou 'fold_1'
    
    # Prioridade de seleção do modelo:
    # 1) saidas/relatorios/metricas_pose.json -> melhor_modelo.path
    # 2) último best.pt por data em modelos/pose/runs
    # 3) pose.model_name do config (pretrained base)
    runs_dir = Path("modelos/pose/runs").resolve()
    model_path = None

    try:
        relatorio_path = Path("saidas/relatorios/metricas_pose.json").resolve()
        if relatorio_path.exists():
            with open(relatorio_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                best_model_path = data.get("melhor_modelo", {}).get("path")
                if best_model_path and Path(best_model_path).exists():
                    model_path = best_model_path
                    logger.info(f"Usando melhor modelo treinado (relatorio): {model_path}")
    except Exception as e:
        logger.warning(f"Erro ao ler metricas_pose.json: {e}. Tentando fallback por runs.")

    if model_path is None:
        candidates = list(runs_dir.rglob("best.pt"))
        if candidates:
            model_path = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
            logger.info(f"Usando modelo encontrado automaticamente (fallback): {model_path}")
        else:
            model_path = config["pose"]["model_name"]  # Fallback para pretrained base
            logger.warning(f"Modelo treinado nao encontrado. Usando base: {model_path}")
        
    dir_saida = Path(config["paths"]["outputs"]) / "inferencias" / "imagens_plotadas"
    
    if desenhar or desenhar_augmentacao:
        dir_saida.mkdir(parents=True, exist_ok=True)
        img_saida = dir_saida / img_path.name
        
    resultado = inferir_keypoints_em_imagem(
        img_path,
        Path(model_path),
        config,
        desenhar,
        dir_saida,
        logger,
        desenhar_augmentacao=desenhar_augmentacao,
        aug_copias=aug_copias,
    )
    
    if desenhar:
        logger.info(f"Inferência concluída. Imagem com plot salva em: {dir_saida / img_path.name}")
    
    print(json.dumps(resultado, default=str, indent=2))

def run_gerar_features(config: dict, logger: logging.Logger) -> Path:
    """
    run_gerar_features: Executa a etapa de geração de features (Fase 2).

    Args:
        config (dict): Dicionário de configuração.
        logger (logging.Logger): Logger configurado.

    Returns:
        Path: Caminho do arquivo CSV de features gerado.
    """
    out_csv = gerar_dataset_features(config, logger)
    return out_csv

def run_analisar_features(config: dict, logger: logging.Logger) -> Path:
    """
    run_analisar_features: Executa analise exploratoria de features da classificacao.

    Args:
        config (dict): Dicionario de configuracao.
        logger (logging.Logger): Logger configurado.

    Returns:
        Path: Caminho do relatorio markdown gerado.
    """
    return analisar_features_exploratorio(config, logger)

def run_classificar_imagem(config: dict, img_path: str, top_k: int, desenhar: bool, logger: logging.Logger) -> None:
    """
    run_classificar_imagem: Executa classificacao de uma imagem unica.
    
    Args:
        config (dict): Dicionario de configuracao.
        img_path (str): Caminho da imagem de entrada.
        top_k (int): Quantidade de classes no ranking top-k.
        desenhar (bool): Se True, salva imagem com predi??o.
        logger (logging.Logger): Logger configurado.
    
    Returns:
        None: Imprime JSON de classificacao e salva imagem de saida quando solicitado.
    """
    from .classificacao.inferencia_classificador import classificar_imagem_unica
    from pathlib import Path
    import json
    
    img_path = Path(img_path)
    if not img_path.exists():
        logger.error(f"Imagem não encontrada: {img_path}")
        return

    resultado = classificar_imagem_unica(config, img_path, top_k, desenhar, logger)
    print(json.dumps(resultado, default=str, indent=2))

if __name__ == "__main__":
    main()

