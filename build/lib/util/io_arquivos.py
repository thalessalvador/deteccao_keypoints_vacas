import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Union

def ler_json(caminho: Union[str, Path]) -> Dict[str, Any]:
    """
    ler_json: Lê um arquivo JSON do disco.

    Args:
        caminho (Union[str, Path]): Caminho completo para o arquivo JSON.

    Returns:
        Dict[str, Any]: O conteúdo do arquivo parseado como dicionário.
    """
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

def salvar_json(dados: Any, caminho: Union[str, Path], indent: int = 2) -> None:
    """
    salvar_json: Salva uma estrutura de dados em formato JSON.

    Args:
        dados (Any): Os dados a serem salvos (deve ser serializável em JSON).
        caminho (Union[str, Path]): Caminho de destino. Diretórios pais serão criados se não existirem.
        indent (int, optional): Nível de indentação do JSON. Padrão é 2.

    Returns:
        None
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=indent, ensure_ascii=False)

def ler_yaml(caminho: Union[str, Path]) -> Dict[str, Any]:
    """
    ler_yaml: Lê um arquivo YAML do disco.

    Args:
        caminho (Union[str, Path]): Caminho completo para o arquivo YAML.

    Returns:
        Dict[str, Any]: O conteúdo do arquivo parseado como dicionário.
    """
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def salvar_yaml(dados: Any, caminho: Union[str, Path]) -> None:
    """
    salvar_yaml: Salva uma estrutura de dados em formato YAML.

    Args:
        dados (Any): Os dados a serem salvos.
        caminho (Union[str, Path]): Caminho de destino. Diretórios pais serão criados se não existirem.

    Returns:
        None
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, 'w', encoding='utf-8') as f:
        yaml.dump(dados, f, allow_unicode=True, default_flow_style=False)

def garantir_diretorio(caminho: Union[str, Path]) -> Path:
    """
    garantir_diretorio: Garante que o diretório pai de um arquivo exista.

    Args:
        caminho (Union[str, Path]): Caminho do arquivo ou diretório. Se tiver extensão, será tratado como arquivo e o pai será criado.

    Returns:
        Path: O objeto Path do diretório garantido.
    """
    p = Path(caminho)
    if p.suffix: # se tem extensão, é arquivo, pega o pai
        p = p.parent
    p.mkdir(parents=True, exist_ok=True)
    return p
