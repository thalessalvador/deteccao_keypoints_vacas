import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .mlp_torch import resolver_device_torch


class SiameseEmbeddingNet(nn.Module):
    """
    Rede MLP para gerar embeddings usados em classificacao por similaridade.

    Parametros:
        input_dim (int): Dimensao de entrada (numero de features).
        embedding_dim (int): Dimensao do embedding final.
        hidden_layers (List[int]): Tamanhos das camadas ocultas.
        activation (str): Funcao de ativacao das camadas ocultas.
        dropout (float): Taxa de dropout entre camadas.

    Retorno:
        None: Modelo PyTorch inicializado.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
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
            layers.append(act.__class__())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h_i
        layers.append(nn.Linear(prev, int(embedding_dim)))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gera embedding normalizado para o batch de entrada.

        Parametros:
            x (torch.Tensor): Tensor de entrada.

        Retorno:
            torch.Tensor: Embeddings L2-normalizados.
        """
        emb = self.backbone(x)
        return nn.functional.normalize(emb, p=2, dim=1)


def _supcon_loss(emb: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Calcula Supervised Contrastive Loss para um batch.

    Parametros:
        emb (torch.Tensor): Embeddings normalizados (N, D).
        y (torch.Tensor): Rotulos inteiros (N,).
        temperature (float): Temperatura da softmax de similaridade.

    Retorno:
        torch.Tensor: Loss escalar. Retorna 0 quando nao ha pares positivos no batch.
    """
    n = emb.size(0)
    if n <= 1:
        return emb.new_tensor(0.0)

    sim = torch.matmul(emb, emb.t()) / max(float(temperature), 1e-8)
    logits_mask = ~torch.eye(n, dtype=torch.bool, device=emb.device)
    exp_sim = torch.exp(sim) * logits_mask
    denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)

    y = y.view(-1, 1)
    pos_mask = (y == y.t()) & logits_mask
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return emb.new_tensor(0.0)

    log_prob = sim - torch.log(denom)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_count.clamp_min(1)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def _gerar_embeddings(
    model: SiameseEmbeddingNet,
    X_np: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Gera embeddings em batches para um array de features.

    Parametros:
        model (SiameseEmbeddingNet): Modelo de embedding.
        X_np (np.ndarray): Features (N, F) em float32.
        device (torch.device): Dispositivo de inferencia.
        batch_size (int): Tamanho de lote.

    Retorno:
        np.ndarray: Embeddings (N, D) em numpy.
    """
    chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for ini in range(0, X_np.shape[0], int(batch_size)):
            fim = min(ini + int(batch_size), X_np.shape[0])
            xb = torch.from_numpy(X_np[ini:fim]).float().to(device)
            emb = model(xb).cpu().numpy()
            chunks.append(emb)
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.vstack(chunks)


def _prototipos_por_classe(emb: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Calcula um prototipo (centroide) por classe no espaco de embedding.

    Parametros:
        emb (np.ndarray): Embeddings de treino (N, D).
        y (np.ndarray): Rotulos inteiros (N,).
        n_classes (int): Numero total de classes.

    Retorno:
        np.ndarray: Matriz de prototipos (C, D) normalizada.
    """
    d = emb.shape[1]
    prot = np.zeros((int(n_classes), int(d)), dtype=np.float32)
    for c in range(int(n_classes)):
        mask = (y == c)
        if np.any(mask):
            prot[c] = emb[mask].mean(axis=0)
    norms = np.linalg.norm(prot, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    prot = prot / norms
    return prot


def _predict_proba_por_prototipo(
    emb: np.ndarray,
    prototipos: np.ndarray,
    temperature: float = 0.15,
) -> np.ndarray:
    """
    Converte distancia para probabilidade via softmax de similaridade por prototipo.

    Parametros:
        emb (np.ndarray): Embeddings das amostras (N, D).
        prototipos (np.ndarray): Prototipos por classe (C, D).
        temperature (float): Temperatura da softmax.

    Retorno:
        np.ndarray: Probabilidades (N, C).
    """
    # Embeddings/prototipos normalizados => dot product ~ similaridade coseno.
    sims = emb @ prototipos.T
    logits = sims / max(float(temperature), 1e-8)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)
    return probs.astype(np.float32)


def _montar_batches_balanceados(
    y_train_np: np.ndarray,
    batch_size: int,
    classes_por_batch: int,
    amostras_por_classe: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Monta batches balanceados por classe para fortalecer pares positivos da loss contrastiva.

    Parametros:
        y_train_np (np.ndarray): Rotulos de treino.
        batch_size (int): Tamanho do lote.
        classes_por_batch (int): Quantidade de classes distintas por batch.
        amostras_por_classe (int): Quantidade de amostras por classe no batch.
        rng (np.random.Generator): Gerador aleatorio.

    Retorno:
        List[np.ndarray]: Lista de vetores de indices (um por batch).
    """
    y_arr = np.asarray(y_train_np, dtype=np.int64).reshape(-1)
    classes_unicas = np.unique(y_arr)
    if classes_unicas.size == 0:
        return []

    classes_por_batch_eff = max(1, int(classes_por_batch))
    amostras_por_classe_eff = max(2, int(amostras_por_classe))

    capacidade = classes_por_batch_eff * amostras_por_classe_eff
    if capacidade > int(batch_size):
        classes_por_batch_eff = max(1, int(batch_size) // amostras_por_classe_eff)
        if classes_por_batch_eff <= 0:
            classes_por_batch_eff = 1
            amostras_por_classe_eff = max(2, int(batch_size))

    tamanho_batch = classes_por_batch_eff * amostras_por_classe_eff
    n_batches = max(1, int(np.ceil(len(y_arr) / max(1, tamanho_batch))))

    idx_por_classe: Dict[int, np.ndarray] = {}
    for c in classes_unicas:
        idx = np.where(y_arr == c)[0]
        if idx.size > 0:
            idx_por_classe[int(c)] = idx
    classes_disponiveis = np.array(sorted(idx_por_classe.keys()), dtype=np.int64)
    if classes_disponiveis.size == 0:
        return []

    batches: List[np.ndarray] = []
    for _ in range(n_batches):
        replace_cls = classes_disponiveis.size < classes_por_batch_eff
        classes_escolhidas = rng.choice(
            classes_disponiveis,
            size=classes_por_batch_eff,
            replace=replace_cls,
        )
        idx_batch: List[int] = []
        for cls in classes_escolhidas:
            idx_cls = idx_por_classe[int(cls)]
            replace_idx = idx_cls.size < amostras_por_classe_eff
            escolhidos = rng.choice(
                idx_cls,
                size=amostras_por_classe_eff,
                replace=replace_idx,
            )
            idx_batch.extend(int(v) for v in escolhidos.tolist())
        rng.shuffle(idx_batch)
        batches.append(np.asarray(idx_batch, dtype=np.int64))
    return batches


def treinar_siamese_torch(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_val_np: np.ndarray,
    params_modelo: Dict[str, Any],
    params_treino: Dict[str, Any],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Treina rede de embedding com Supervised Contrastive Loss e early stopping por F1.

    Parametros:
        X_train_np (np.ndarray): Features de treino normalizadas.
        y_train_np (np.ndarray): Rotulos de treino.
        X_val_np (np.ndarray): Features de validacao normalizadas.
        y_val_np (np.ndarray): Rotulos de validacao.
        params_modelo (Dict[str, Any]): Parametros de arquitetura.
        params_treino (Dict[str, Any]): Parametros de treino.
        device (torch.device): Dispositivo de treino.
        seed (int): Semente aleatoria.

    Retorno:
        Dict[str, Any]: Modelo final, prototipos, metricas e historico de treino.
    """
    from sklearn.metrics import f1_score, accuracy_score

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = SiameseEmbeddingNet(
        input_dim=int(params_modelo["input_dim"]),
        embedding_dim=int(params_modelo["embedding_dim"]),
        hidden_layers=[int(v) for v in params_modelo.get("hidden_layers", [128, 64])],
        activation=str(params_modelo.get("activation", "relu")),
        dropout=float(params_modelo.get("dropout", 0.0)),
    ).to(device)

    lr = float(params_treino.get("lr", 1e-3))
    wd = float(params_treino.get("weight_decay", 1e-4))
    batch_size = int(params_treino.get("batch_size", 128))
    max_epochs = int(params_treino.get("max_epochs", 300))
    patience = int(params_treino.get("patience", 30))
    min_delta = float(params_treino.get("min_delta", 0.0))
    temperature = float(params_treino.get("temperature", 0.15))
    batch_balanceado = bool(params_treino.get("batch_balanceado", True))
    classes_por_batch = int(params_treino.get("classes_por_batch", 16))
    amostras_por_classe = int(params_treino.get("amostras_por_classe", 4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)

    x_train_t = torch.from_numpy(X_train_np).float()
    y_train_t = torch.from_numpy(y_train_np).long()

    historico: Dict[str, List[float]] = {
        "loss_train": [],
        "loss_val": [],
        "acc_val": [],
        "f1_val": [],
    }

    melhor_f1 = -1.0
    melhor_loss_val = float("inf")
    melhor_acc = 0.0
    melhor_epoca = 0
    melhor_state: Optional[Dict[str, torch.Tensor]] = None
    melhor_prototipos: Optional[np.ndarray] = None
    sem_melhora = 0

    n_classes = int(len(np.unique(y_train_np)))

    for epoca in range(1, max_epochs + 1):
        model.train()
        perda_epoca = 0.0
        n_batches = 0

        if batch_balanceado:
            batches_idx = _montar_batches_balanceados(
                y_train_np=y_train_np,
                batch_size=batch_size,
                classes_por_batch=classes_por_batch,
                amostras_por_classe=amostras_por_classe,
                rng=rng,
            )
            if not batches_idx:
                batches_idx = [rng.permutation(X_train_np.shape[0]).astype(np.int64)]
        else:
            idx = rng.permutation(X_train_np.shape[0])
            batches_idx = []
            for ini in range(0, len(idx), batch_size):
                fim = min(ini + batch_size, len(idx))
                batches_idx.append(idx[ini:fim].astype(np.int64))

        for b_idx in batches_idx:
            if b_idx.size <= 1:
                continue
            xb = x_train_t[b_idx].to(device)
            yb = y_train_t[b_idx].to(device)

            optimizer.zero_grad()
            emb = model(xb)
            loss = _supcon_loss(emb, yb, temperature=temperature)
            loss.backward()
            optimizer.step()

            perda_epoca += float(loss.detach().cpu().item())
            n_batches += 1

        loss_train = perda_epoca / max(1, n_batches)

        emb_train = _gerar_embeddings(model, X_train_np, device=device, batch_size=batch_size)
        prototipos = _prototipos_por_classe(emb_train, y_train_np, n_classes=n_classes)

        emb_val = _gerar_embeddings(model, X_val_np, device=device, batch_size=batch_size)
        probs_val = _predict_proba_por_prototipo(emb_val, prototipos, temperature=temperature)
        pred_val = np.argmax(probs_val, axis=1)
        acc_val = float(accuracy_score(y_val_np, pred_val))
        f1_val = float(f1_score(y_val_np, pred_val, average="macro", zero_division=0))

        # Reaproveita a loss contrastiva como valor de validacao para rastreio.
        with torch.no_grad():
            xv = torch.from_numpy(X_val_np).float().to(device)
            yv = torch.from_numpy(y_val_np).long().to(device)
            loss_val = float(_supcon_loss(model(xv), yv, temperature=temperature).cpu().item())

        historico["loss_train"].append(loss_train)
        historico["loss_val"].append(loss_val)
        historico["acc_val"].append(acc_val)
        historico["f1_val"].append(f1_val)

        if f1_val > (melhor_f1 + min_delta):
            melhor_f1 = f1_val
            melhor_loss_val = loss_val
            melhor_acc = acc_val
            melhor_epoca = epoca
            melhor_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            melhor_prototipos = prototipos.copy()
            sem_melhora = 0
        else:
            sem_melhora += 1

        if sem_melhora >= patience:
            break

    if melhor_state is not None:
        model.load_state_dict(melhor_state)
    if melhor_prototipos is None:
        emb_train = _gerar_embeddings(model, X_train_np, device=device, batch_size=batch_size)
        melhor_prototipos = _prototipos_por_classe(emb_train, y_train_np, n_classes=n_classes)

    return {
        "model": model,
        "prototipos": melhor_prototipos,
        "best_f1": float(melhor_f1),
        "best_val_loss": float(melhor_loss_val),
        "best_acc": float(melhor_acc),
        "best_epoch": int(melhor_epoca),
        "n_epochs_executadas": int(len(historico["loss_train"])),
        "historico": historico,
    }


def salvar_checkpoint_siamese_torch(
    model_path: Path,
    scaler_path: Path,
    model: SiameseEmbeddingNet,
    scaler: StandardScaler,
    prototipos: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """
    Salva checkpoint do classificador siames em PyTorch.

    Parametros:
        model_path (Path): Caminho do arquivo `.pt`.
        scaler_path (Path): Caminho do scaler `.joblib`.
        model (SiameseEmbeddingNet): Modelo treinado.
        scaler (StandardScaler): Scaler treinado.
        prototipos (np.ndarray): Prototipos por classe.
        metadata (Dict[str, Any]): Metadados de arquitetura/treino.

    Retorno:
        None: Arquivos persistidos em disco.
    """
    payload = dict(metadata)
    payload["state_dict"] = model.state_dict()
    payload["prototipos"] = np.asarray(prototipos, dtype=np.float32)
    torch.save(payload, model_path)
    joblib.dump(scaler, scaler_path)


def carregar_checkpoint_siamese_torch(
    model_path: Path,
    scaler_path: Path,
    device_cfg: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Carrega checkpoint do classificador siames para inferencia/avaliacao.

    Parametros:
        model_path (Path): Caminho do checkpoint `.pt`.
        scaler_path (Path): Caminho do scaler `.joblib`.
        device_cfg (Optional[str]): Configuracao de dispositivo.
        logger (Optional[logging.Logger]): Logger para mensagens.

    Retorno:
        Dict[str, Any]: Artefato completo com modelo, scaler, prototipos e metadados.
    """
    device = resolver_device_torch(device_cfg, logger=logger)
    # PyTorch 2.6 mudou o padrao de torch.load para weights_only=True.
    # Nosso checkpoint inclui metadados + arrays numpy, entao precisamos
    # carregar com weights_only=False para compatibilidade.
    try:
        payload = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibilidade com versoes antigas do PyTorch sem argumento weights_only.
        payload = torch.load(model_path, map_location=device)
    scaler: StandardScaler = joblib.load(scaler_path)
    model = SiameseEmbeddingNet(
        input_dim=int(payload["input_dim"]),
        embedding_dim=int(payload["embedding_dim"]),
        hidden_layers=[int(v) for v in payload.get("hidden_layers", [128, 64])],
        activation=str(payload.get("activation", "relu")),
        dropout=float(payload.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    prototipos = np.asarray(payload["prototipos"], dtype=np.float32)
    return {
        "model": model,
        "scaler": scaler,
        "device": device,
        "prototipos": prototipos,
        "metadata": payload,
    }


def predict_proba_siamese_torch(
    X: Union[pd.DataFrame, np.ndarray],
    artefato: Dict[str, Any],
    batch_size: int = 1024,
    temperature: Optional[float] = None,
) -> np.ndarray:
    """
    Prediz probabilidade por similaridade com prototipos no espaco de embedding.

    Parametros:
        X (Union[pd.DataFrame, np.ndarray]): Features de entrada.
        artefato (Dict[str, Any]): Saida de `carregar_checkpoint_siamese_torch`.
        batch_size (int): Tamanho de batch de inferencia.
        temperature (Optional[float]): Temperatura opcional para softmax.

    Retorno:
        np.ndarray: Matriz de probabilidades (N, C).
    """
    if isinstance(X, pd.DataFrame):
        arr = X.values.astype(np.float32)
    else:
        arr = np.asarray(X, dtype=np.float32)

    scaler: StandardScaler = artefato["scaler"]
    Xn = scaler.transform(arr).astype(np.float32)
    model: SiameseEmbeddingNet = artefato["model"]
    device: torch.device = artefato["device"]
    prototipos: np.ndarray = artefato["prototipos"]
    temp = float(
        temperature
        if temperature is not None
        else artefato.get("metadata", {}).get("temperature", 0.15)
    )
    emb = _gerar_embeddings(model, Xn, device=device, batch_size=batch_size)
    return _predict_proba_por_prototipo(emb, prototipos, temperature=temp)
