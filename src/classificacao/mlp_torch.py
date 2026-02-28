import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class MLPDinamicoTorch(nn.Module):
    """
    Rede MLP totalmente conectada com arquitetura dinamica.

    Parametros:
        input_dim (int): Quantidade de features de entrada.
        num_classes (int): Quantidade de classes de saida.
        hidden_layers (List[int]): Lista com tamanhos das camadas ocultas.
        activation (str): Funcao de ativacao das camadas ocultas.
        dropout (float): Taxa de dropout aplicada apos cada camada oculta.

    Retorno:
        None: Objeto de modelo PyTorch inicializado.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        act_map: Dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        }
        act = act_map.get(str(activation).lower(), nn.ReLU())

        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_layers:
            h_i = int(h)
            layers.append(nn.Linear(prev, h_i))
            layers.append(act.__class__())  # cria nova instancia da ativacao
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h_i
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa passagem direta do modelo.

        Parametros:
            x (torch.Tensor): Batch de entrada.

        Retorno:
            torch.Tensor: Logits de saida.
        """
        return self.net(x)


def resolver_device_torch(device_cfg: Optional[str], logger: Optional[logging.Logger] = None) -> torch.device:
    """
    Resolve o dispositivo PyTorch a partir da configuracao textual.

    Parametros:
        device_cfg (Optional[str]): Valor de configuracao (ex.: "cuda", "cpu", "0", "1").
        logger (Optional[logging.Logger]): Logger para mensagens informativas.

    Retorno:
        torch.device: Dispositivo final escolhido para execucao.
    """
    cfg = "cuda" if device_cfg is None else str(device_cfg).lower().strip()
    if cfg in ("cuda", "gpu", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if logger is not None:
            logger.warning("CUDA nao disponivel. Usando CPU para MLP Torch.")
        return torch.device("cpu")
    if cfg.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cfg}")
        if logger is not None:
            logger.warning("Indice de GPU solicitado, mas CUDA nao disponivel. Usando CPU.")
        return torch.device("cpu")
    if cfg.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(cfg)
        if logger is not None:
            logger.warning("Dispositivo CUDA especificado sem suporte local. Usando CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def salvar_checkpoint_mlp_torch(
    model_path: Path,
    scaler_path: Path,
    model: MLPDinamicoTorch,
    scaler: StandardScaler,
    metadata: Dict[str, Any],
) -> None:
    """
    Salva checkpoint do modelo MLP Torch e scaler associado.

    Parametros:
        model_path (Path): Caminho do arquivo `.pt`.
        scaler_path (Path): Caminho do scaler `.joblib`.
        model (MLPDinamicoTorch): Modelo treinado.
        scaler (StandardScaler): Scaler treinado com dados de treino.
        metadata (Dict[str, Any]): Metadados de arquitetura e hiperparametros.

    Retorno:
        None: Arquivos de modelo e scaler persistidos em disco.
    """
    payload = dict(metadata)
    payload["state_dict"] = model.state_dict()
    torch.save(payload, model_path)
    joblib.dump(scaler, scaler_path)


def carregar_checkpoint_mlp_torch(
    model_path: Path,
    scaler_path: Path,
    device_cfg: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Carrega modelo MLP Torch e scaler para inferencia.

    Parametros:
        model_path (Path): Caminho do checkpoint `.pt`.
        scaler_path (Path): Caminho do scaler `.joblib`.
        device_cfg (Optional[str]): Configuracao de dispositivo.
        logger (Optional[logging.Logger]): Logger para mensagens.

    Retorno:
        Dict[str, Any]: Dicionario com modelo, scaler, device e metadados.
    """
    device = resolver_device_torch(device_cfg, logger=logger)
    payload = torch.load(model_path, map_location=device)
    scaler: StandardScaler = joblib.load(scaler_path)
    model = MLPDinamicoTorch(
        input_dim=int(payload["input_dim"]),
        num_classes=int(payload["num_classes"]),
        hidden_layers=[int(v) for v in payload["hidden_layers"]],
        activation=str(payload.get("activation", "relu")),
        dropout=float(payload.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "scaler": scaler,
        "device": device,
        "metadata": payload,
    }


def predict_proba_mlp_torch(
    X: Union[pd.DataFrame, np.ndarray],
    artefato: Dict[str, Any],
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Calcula probabilidades de classe para um conjunto de amostras.

    Parametros:
        X (Union[pd.DataFrame, np.ndarray]): Matriz de features.
        artefato (Dict[str, Any]): Saida de `carregar_checkpoint_mlp_torch`.
        batch_size (int): Tamanho do batch para inferencia.

    Retorno:
        np.ndarray: Matriz de probabilidades com formato (N, C).
    """
    if isinstance(X, pd.DataFrame):
        arr = X.values.astype(np.float32)
    else:
        arr = np.asarray(X, dtype=np.float32)
    scaler: StandardScaler = artefato["scaler"]
    Xn = scaler.transform(arr).astype(np.float32)
    model: MLPDinamicoTorch = artefato["model"]
    device: torch.device = artefato["device"]

    probs_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for ini in range(0, Xn.shape[0], int(batch_size)):
            fim = min(ini + int(batch_size), Xn.shape[0])
            xb = torch.from_numpy(Xn[ini:fim]).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_chunks.append(probs)
    if not probs_chunks:
        return np.zeros((0, int(artefato["metadata"]["num_classes"])), dtype=np.float32)
    return np.vstack(probs_chunks)
