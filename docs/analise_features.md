# Analise Exploratoria de Dados (EDA)

## Relatorio de Analise Exploratoria

### 1. Visao geral do dataset

O conjunto de dados esta robusto para classificacao de 30 individuos (vacas),
com boa cobertura de variacoes devido ao uso de augmentacao.

- Volume de dados: **14945 amostras**
- Classes: **30**
- Features numericas: **50**
- Estrategia de dados: **1495 instancias reais** e **13450 instancias aumentadas**
- Split externo registrado no CSV: **14795 treino** e **150 teste**

Interpretacao: ha predominio de dados augmentados no treino, o que tende a
melhorar robustez a variacoes de pose/ruido, desde que as transformacoes
permaneçam biologicamente plausiveis.

### 2. Qualidade e limpeza dos dados

- Missing values: **0% em todas as features**
- Features constantes detectadas: **sc_withers_x** e **sc_withers_y**

Interpretacao: a qualidade tabular esta alta (sem lacunas). As features
constantes nao trazem ganho discriminativo e podem permanecer desativadas no
treino.

### 3. Correlacao e redundancia

As correlacoes mais altas mostram redundancia geometrica relevante:

- `angulo_withers_back_tail_head` x `desvio_coluna_back_norm`: **-0.9970**
- `area_triangulo_torax_norm` x `desvio_coluna_back_norm`: **0.9659**
- `area_poligono_pelvico_norm` x `indice_triangulo_traseiro`: **0.9573**
- `indice_robustez` x `pca_excentricidade`: **-0.9351**

Interpretacao: varias features estao descrevendo o mesmo fenomeno morfologico
por formulacoes diferentes. Isso e esperado em engenharia de features
geometricas. Em modelos lineares, essa redundancia pode prejudicar; em modelos
nao lineares, tende a ser menos critica.

### 4. Importancia das features (o que mais diferencia os individuos)

Pelo ranking combinado (ANOVA + Mutual Information), as mais relevantes foram:

1. `bbox_aspect_ratio`
2. `sc_hook_up_y`
3. `dist_tail_head_pin_up`
4. `dist_back_hook_up`
5. `pca_excentricidade`
6. `razao_dist_hip_hook_up_por_dist_hook_up_pin_up`
7. `dist_hip_tail_head`
8. `indice_robustez`

Interpretacao:

- features de forma global (`bbox_aspect_ratio`, `pca_excentricidade`,
  `indice_robustez`) aparecem forte;
- medidas da regiao pelvica e dorsal seguem importantes;
- coordenadas relativas no eixo Y (shape-context) contribuem para diferenciar
  padrao morfologico individual.

### 5. Padroes significativos e insights

- A regiao traseira/pelvica concentra varias features relevantes, coerente com
  variacao anatomica visivel entre animais.
- O PCA 2D e util para inspeção visual, mas nao deve ser usado isoladamente
  para concluir separabilidade completa entre 30 classes.
- O conjunto atual sugere boa capacidade descritiva das features, com espaco
  para enxugamento controlado (ablation) sem perder desempenho.

