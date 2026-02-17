import logging
import sys
from pathlib import Path

def configurar_logger(nome: str = "projeto_vacas", nivel: int = logging.INFO, arquivo_log: str = "saidas/logs/app.log") -> logging.Logger:
    """
    configurar_logger: Configura e retorna um logger padronizado para a aplicação.

    Configura handlers para saída padrão (stdout) e arquivo, formatando as mensagens
    com timestamp, nome do logger, nível e mensagem. Evita duplicação de handlers
    se o logger já estiver configurado.

    Args:
        nome (str, optional): Nome do logger a ser recuperado/criado. Padrão é "projeto_vacas".
        nivel (int, optional): Nível mínimo de log (ex: logging.INFO, logging.DEBUG). Padrão é logging.INFO.
        arquivo_log (str, optional): Caminho para o arquivo de log. Padrão é "saidas/logs/app.log".

    Returns:
        logging.Logger: Instância do logger configurado.
    """
    logger = logging.getLogger(nome)
    logger.setLevel(nivel)
    
    # Evita duplicidade de handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler de Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler de Arquivo
    caminho_log = Path(arquivo_log)
    caminho_log.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(caminho_log, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
