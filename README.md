# Projeto Vacas — Pose (YOLOv8) + Identificação (XGBoost)

## Visão geral
Este repositório implementa um pipeline em 3 fases:

1. **Fase 1 — Pose/Keypoints (YOLOv8 Pose)**  
   Converte anotações do Label Studio em formato YOLOv8 Pose, treina um modelo para detectar **bbox de vaca** e **8 keypoints**, e disponibiliza inferência (JSON e opcionalmente imagem desenhada).

2. **Fase 2 — Geração de features (CSV)**  
   Usa o modelo de pose para extrair keypoints do `dataset_classificacao` e gerar um CSV com features geométricas por imagem (90% de cada vaca).  
   Inclui **seleção robusta da instância-alvo** na imagem (para o caso de existir uma segunda vaca parcialmente no frame).

3. **Fase 3 — Classificação da vaca (XGBoost)**  
   Treina um classificador tabular (**XGBoost por padrão; RandomForest baseline opcional**) e avalia no “caso real” (10%), gerando **matriz de confusão** e métricas globais. Também suporta inferência em imagem única com top-k.

---

## Requisitos
- Python 3.10+ (recomendado)
- Dependências principais:
  - ultralytics
  - opencv-python
  - numpy, pandas
  - scikit-learn
  - matplotlib
  - pyyaml
  - xgboost (padrão)
- GPU NVIDIA (opcional, recomendado): qualquer RTX com CUDA compatível.

---

## Instalação

### 1) Criar ambiente virtual
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2) Instalar dependências e o projeto
```bash
pip install -e .
```
Isso instalará as dependências listadas no `pyproject.toml` e o próprio pacote `src` em modo editável.

### 3) Configuração de GPU (CUDA) vs CPU
Este projeto roda tanto em CPU quanto em GPU. Para usar GPU:

1. Atualize o **driver NVIDIA**.
2. Verifique se o PyTorch reconhece sua GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```
Se retornar `True` e o nome da GPU, tudo pronto.

Se retornar `False no-cuda`:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(Ajuste `cu121` para a versão do seu CUDA, ex: cu118, cu124)*

#### Troubleshooting e GPUs Novas (RTX série 50)
- **CUDA mismatch**: reinstale com a versão correta.
- **Erro `cudaErrorNotSupported` ou warning de arquitetura**: Instale o build **Nightly** (ex: `cu128`):
```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
- **Out of memory (VRAM 8GB)**: Reduza `pose.batch` e `pose.imgsz` no `config.yaml`.

---

## Estrutura de pastas (alto nível)
```text
dados/raw/
  dataset_keypoints/
  dataset_classificacao/
dados/processados/
  yolo_pose/
  classificacao/
modelos/
saidas/
src/
config/
```

---

## Formato dos datasets

### dataset_keypoints
```text
dados/raw/dataset_keypoints/
  <anotador>/
    ...imagens...
    Key_points/
      1.json
      2.json
      ...
```
- Pode haver múltiplas vacas por imagem.
- Label Studio tem visibilidade **visível/oculto**.  
  Pontos fora do quadro **não aparecem** na anotação (tratados como inexistentes).
- Cenário real: filmagem **de cima** (não há uma vaca passando por cima de outra).  
  Pode haver **overlap leve de bordas** dos bboxes; por isso, a associação keypoint↔bbox usa tolerância (2–5 px).

### dataset_classificacao
```text
dados/raw/dataset_classificacao/
  <cow_number>/
    *.jpg
```
- ~50 imagens por vaca.
- Split 90/10 por vaca.

---

## Formato YOLOv8 Pose gerado
- Saída: `dados/processados/yolo_pose/images` e `labels`.
- Uma linha por vaca/instância:
```text
0 xc yc w h k1x k1y k1v ... k8x k8y k8v
```
- `v`: 2 (visível), 1 (oculto), 0 (inexistente).

---

## Configuração
Edite `config/config.yaml`.

Principais parâmetros:
- `pose.imgsz`, `pose.batch`, `pose.epochs`, `pose.device`
- `pose.k_folds`, `pose.estrategia_validacao`
- `pose.usar_data_augmentation` e `pose.augmentacao`
- `classificacao.features_selecionadas`, `classificacao.modelo_padrao`

---

## Data augmentation (YOLOv8 nativo)
A Fase 1 utiliza augmentations nativas do Ultralytics/YOLOv8 (sem Albumentations).

### Política de flip horizontal (`fliplr`)
- **Padrão:** `fliplr=0.0` (desligado) para evitar efeitos de orientação nos recursos geométricos.
- Se você ativar `fliplr>0`, **obrigatório** ativar `classificacao.normalizar_orientacao=true` para normalizar os keypoints no espaço do bbox antes de calcular features.

Exemplo:
```yaml
pose:
  usar_data_augmentation: true
  augmentacao:
    degrees: 5.0
    fliplr: 0.0
    flipud: 0.0
    mosaic: 0.7
    mixup: 0.1
classificacao:
  normalizar_orientacao: false
```

Recomendações:
- manter `degrees` baixo (ex.: 5–10)
- `flipud=0.0`
- `mosaic/mixup` moderados

---

## Seleção da instância-alvo (Fase 2)
Mesmo com filmagem de cima, pode existir uma segunda vaca parcialmente no frame.

A seleção da instância-alvo usa:
- **confiança mínima** (`conf_min`)
- **área do bbox**
- **proximidade do centro do bbox ao centro da imagem** (prioriza o alvo principal)

Se nenhuma instância passar `conf_min`, a imagem é **descartada do treino** e registrada em relatório.

---

## Treino do classificador (XGBoost) — early stopping
O treino tabular cria uma **validação interna** (ex.: 80/20 dentro do treino 90%) e utiliza:
- `early_stopping_rounds`

Isso reduz overfitting quando `n_estimators` é alto (ex.: 800).

---

## Uso via CLI

### Ajuda
```bash
python -m src.cli --help
```

### Pipeline completo (até matriz de confusão final)
```bash
python -m src.cli pipeline-completo --config config/config.yaml
```

### Inferir pose (JSON) e desenhar pontos
```bash
python -m src.cli inferir-pose --config config/config.yaml --imagem CAMINHO_IMAGEM --desenhar
```

### Classificar vaca (top-k) e desenhar keypoints
```bash
python -m src.cli classificar-imagem --config config/config.yaml --imagem CAMINHO_IMAGEM --top-k 5 --desenhar
```

---

## Saídas e métricas
- Pose:
  - `saidas/relatorios/metricas_pose.json`
- Classificação:
  - `saidas/relatorios/metricas_classificacao.json`
  - `saidas/relatorios/matriz_confusao.png`
  - `saidas/relatorios/matriz_confusao.csv`

---

## Reprodutibilidade
- Seeds configuráveis em `config.yaml`.
- Logs em `saidas/logs/app.log`.

---

## Licença
Definir conforme necessidade do projeto.
