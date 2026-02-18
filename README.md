# Projeto Vacas — Pose (YOLO, yolo26n-pose) + Identificação (XGBoost/CatBoost/RF)

## Visão geral
Este repositório implementa um pipeline em 3 fases:

1. **Fase 1 — Pose/Keypoints (YOLO Pose)**  
   Converte anotações do Label Studio em formato YOLO Pose, treina um modelo para detectar **bbox de vaca** e **8 keypoints**, e disponibiliza inferência (JSON e opcionalmente imagem desenhada).

2. **Fase 2 — Geração de features (CSV)**  
   Usa o modelo de pose para extrair keypoints do `dataset_classificacao` e gerar um CSV com features geométricas por imagem (90% de cada vaca).  
   Inclui **seleção robusta da instância-alvo** na imagem (para o caso de existir uma segunda vaca parcialmente no frame).

3. **Fase 3 — Classificação da vaca (XGBoost/CatBoost/RF)**  
   Treina um classificador tabular (**XGBoost por padrão; CatBoost e RandomForest como alternativas**) e avalia no “caso real” (10%), gerando **matriz de confusão** e métricas globais. Também suporta inferência em imagem única com top-k.

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
  - xgboost / catboost (classificação tabular)
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
      1 (arquivo JSON, pode não ter extensão .json)
      2
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

## Formato YOLO Pose gerado
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
- `pose.model_name` (atual: `yolo26n-pose.pt`)
- `pose.imgsz`, `pose.batch`, `pose.epochs`, `pose.device`
- `pose.k_folds`, `pose.estrategia_validacao`
- `pose.usar_data_augmentation` e `pose.augmentacao`
- `classificacao.modelo_padrao` (`xgboost`, `catboost` ou `sklearn_rf`)
- `classificacao.features.selecionadas`
- `classificacao.usar_data_augmentation`, `classificacao.augmentacao_keypoints`
- `classificacao.validacao_interna.usar_apenas_real` (validação interna só com instâncias reais)

---

## Detalhes das Features Geométricas (Fase 2)
O sistema extrai um conjunto robusto de features visando invariância a escala, rotação e translação (posição da vaca na imagem).

### 1. Features Básicas (BBox)
- `bbox_aspect_ratio`: Razão largura/altura do bounding box (forma geral).
- `bbox_area_norm`: Área do bbox normalizada pela área da imagem (tamanho relativo).

### 2. Razões de Distâncias
Relações adimensionais entre segmentos corporais, capturando proporções anatômicas independente do zoom:
- Ex: `razao_dist_hip_hook_up_por_dist_hook_up_pin_up` (Proporção do quadril).

### 3. Ângulos
Ângulos em graus formados por *trios* de keypoints, essenciais para capturar postura e angulação óssea:
- Ex: `angulo_hook_up_hip_hook_down` (Abertura pélvica).

### 4. Áreas de Polígonos (Normalizadas)
Medidas de "volume" de regiões específicas:
- `area_poligono_pelvico_norm`: Área do trapézio formado pelos ganchos (`hooks`) e pinos (`pins`).
- `area_triangulo_torax_norm`: Área do triângulo frontal (`withers`, `back`, `hip`).

### 5. Índices de Conformação
- `indice_robustez`: Largura dos ganchos dividida pelo comprimento do corpo (`withers` -> `tail`).
- `indice_triangulo_traseiro`: Área do triângulo traseiro relativa ao bbox.

### 6. Curvatura da Coluna
Indica desvios laterais (escoliose/postura):
- `desvio_coluna_back_norm`: Distância perpendicular das costas (`back`) em relação à linha reta teórica que liga cernelha à cauda.

### 7. Shape Context (Coordenadas Relativas)
Transformação geométrica que re-projeta todos os pontos (`sc_*_x`, `sc_*_y`) em um sistema onde:
- Origem (0,0) é a Cernelha (`withers`).
- Eixo X é alinhado com a Cauda (`tail_head`).
Isso elimina o efeito da rotação da vaca na imagem, permitindo usar a "forma" pura.

### 8. Excentricidade (PCA)
- `pca_excentricidade`: Razão entre os autovalores principais da distribuição de pontos (indica se a vaca é mais "alongada" ou "arredondada").

### 9. Features adicionais (distâncias/ângulos)
Incluídas para aproximar variáveis usadas em artigos de conformação:
- `dist_hip_tail_head`
- `dist_tail_head_pin_up`
- `dist_back_hook_up`
- `razao_largura_hooks_por_largura_pins`
- `angulo_hook_up_pin_up_tail_head`

---

## Data augmentation (YOLO nativo)
A Fase 1 utiliza augmentations nativas do Ultralytics/YOLO (sem Albumentations).

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

## Data augmentation da classificação (keypoints)
Na Fase 2 (`gerar-features`), o pipeline pode gerar amostras sintéticas a partir dos keypoints inferidos:
- A linha **real** é sempre gerada.
- Cópias com ruído gaussiano em `(x,y)` são geradas **somente para split de treino**.
- O ruído é escalado pelo tamanho do bbox (`noise_std_xy`).
- Keypoints com confiança abaixo de `conf_min_keypoint` não recebem ruído.

Parâmetros em `config.yaml`:
- `classificacao.usar_data_augmentation`
- `classificacao.augmentacao_keypoints.habilitar`
- `classificacao.augmentacao_keypoints.n_copias`
- `classificacao.augmentacao_keypoints.noise_std_xy`
- `classificacao.augmentacao_keypoints.conf_min_keypoint`
- `classificacao.augmentacao_keypoints.clip_coords`
- `classificacao.augmentacao_keypoints.deterministico`
- `classificacao.augmentacao_keypoints.seed`

O CSV de features inclui metadados:
- `origem_instancia` (`real` ou `augmentation`)
- `is_aug`, `aug_id`, `split_instancia`

---

## Seleção da instância-alvo (Fase 2)
Mesmo com filmagem de cima, pode existir uma segunda vaca parcialmente no frame.

A seleção da instância-alvo usa:
- **confiança mínima** (`conf_min`)
- **área do bbox**
- **proximidade do centro do bbox ao centro da imagem** (prioriza o alvo principal)

Se nenhuma instância passar `conf_min`, a imagem é **descartada do treino** e registrada em relatório.

---

## Treino do classificador (XGBoost/CatBoost/RF) — early stopping
O treino tabular cria uma **validação interna** (ex.: 80/20 dentro do treino 90%) e utiliza:
- `early_stopping_rounds`

Para evitar vazamento entre cópias augmentadas da mesma imagem, a validação interna é feita por **grupo `arquivo`**.
Opcionalmente, é possível filtrar a validação interna para usar apenas instâncias reais com
`classificacao.validacao_interna.usar_apenas_real=true`.

Isso reduz overfitting quando `n_estimators` é alto (ex.: 800).

---

## Uso via CLI

O script `src/cli.py` é o ponto central de execução.

**Nota sobre Configuração:**
O argumento `--config` é **global e opcional**. Se não informado, usa `config/config.yaml`.
Para usar outro arquivo, passe-o **antes** do subcomando:
```bash
python -m src.cli --config meu_config.yaml <COMANDO>
```

### Comandos Disponíveis

#### 1. Pré-processamento (Fase 1)
Converte anotações do Label Studio (JSON) para formato YOLO Pose e cria `dataset.yaml`:
```bash
python -m src.cli preprocessar-pose
```

#### 2. Treinar Pose (Fase 1)
Inicia o treinamento do YOLO Pose (usando k-fold ou split simples definido no config):
```bash
python -m src.cli treinar-pose
```
Estratégias de validação suportadas em `pose.estrategia_validacao`:
- `kfold_misturado`
- `groupkfold_por_sessao` (recomendado para reduzir vazamento entre frames/sessões semelhantes)
- `groupkfold_por_anotador` (proxy por origem/prefixo no nome do arquivo)

#### 3. Inferir Pose (Teste Fase 1)
Roda o modelo de pose em uma imagem e opcionalmente desenha o esqueleto:
```bash
python -m src.cli inferir-pose --imagem "caminho/para/imagem.jpg" --desenhar
```
*A saída será salva em `saidas/inferencias/imagens_plotadas`.*

#### 4. Gerar Features (Fase 2)
Processa todas as imagens de `dataset_classificacao`, extrai keypoints e calcula as features geométricas (CSV):
```bash
python -m src.cli gerar-features
```

#### 5. Treinar Classificador (Fase 3)
Treina o modelo definido em `classificacao.modelo_padrao` e salva os artefatos do classificador (`xgboost_model.json` ou `catboost_model.cbm` ou `rf_model.joblib`, além do encoder):
```bash
python -m src.cli treinar-classificador
```

#### 6. Avaliar Classificador (Fase 3)
Avalia o modelo treinado no conjunto de teste (10% isolado por vaca) e gera matriz de confusão:
```bash
python -m src.cli avaliar-classificador
```

#### 7. Classificar Imagem (End-to-End)
Executa o fluxo completo para uma nova imagem:
1. Detecta pose (YOLO).
2. Extrai features.
3. Classifica (XGBoost).
```bash
python -m src.cli classificar-imagem --imagem "cam/para/img.jpg" --top-k 3 --desenhar
```
*Gera JSON com probabilidades e salva imagem com predição.*

> **Importante:** Para evitar vazamento de dados (avaliar uma imagem que o modelo já viu no treino), utilize imagens listadas em `dados/processados/classificacao/splits/teste_10pct.txt`. Este arquivo contém os **nomes dos arquivos** (`arquivo.jpg`) reservados para teste.

#### 8. Pipeline Completo
Executa todas as etapas em sequência (útil para reprodução total):
```bash
python -m src.cli pipeline-completo
```
---

## Saídas e métricas
- Pose:
  - `saidas/relatorios/metricas_pose.json`
- Classificação:
  - `saidas/relatorios/metricas_classificacao.json`
  - `saidas/relatorios/metricas_classificacao_treino.json`
  - `saidas/relatorios/matriz_confusao.png`
  - `saidas/relatorios/matriz_confusao.csv`
  - `saidas/relatorios/metricas_por_classe.png`
  - `saidas/relatorios/confianca_corretas_vs_incorretas.png`
  - `saidas/relatorios/cobertura_vs_acuracia.png`
  - `saidas/relatorios/xgb_curva_mlogloss.png`
  - `saidas/relatorios/xgb_curva_merror.png`
  - `saidas/relatorios/xgb_gap_mlogloss.png`
  - `saidas/relatorios/xgb_gap_merror.png`
  - `saidas/relatorios/xgb_importancia_gain_topn.png`

---

## Reprodutibilidade
- Seeds configuráveis em `config.yaml`.
- Logs em `saidas/logs/app.log`.

---

## Licença
Definir conforme necessidade do projeto.
