# Analise Exploratória de Dados (EDA)

Este documento descreve como rodar a analise exploratoria das features da Fase 2
sem alterar o pipeline principal de treino/inferencia.

## Objetivo

Atender ao requisito de analise descritiva/exploratoria das features usadas na
identificacao das vacas, com foco em:

- padroes de distribuicao;
- qualidade dos dados (missing);
- correlacoes e redundancia entre features;
- separabilidade entre classes;
- ranking de relevancia das features.

## Como executar

```bash
python -m src.cli analisar-features
```

Pre-requisito:

- `dados/processados/classificacao/features/features_completas.csv` deve existir
  (gerado por `python -m src.cli gerar-features`).

## Saidas geradas

Diretorio: `saidas/analise_features/`

- `relatorio_eda.md`
- `resumo_eda.json`
- `eda_distribuicao_classes.png`
- `eda_missing_top25.png`
- `eda_heatmap_correlacao.png`
- `eda_top25_importancia.png`
- `eda_pca_2d_top12.png`
- `eda_missing_por_feature.csv`
- `eda_correlacao.csv`
- `eda_importancia_features.csv`
- `eda_top_correlacoes.csv`

## Interpretacao recomendada

1. Comece por `eda_distribuicao_classes.png` para verificar balanceamento.
2. Valide qualidade dos dados em `eda_missing_por_feature.csv`.
3. Revise redundancias em `eda_heatmap_correlacao.png` e `eda_top_correlacoes.csv`.
4. Use `eda_importancia_features.csv` para orientar ablation/selecionar features.
5. Verifique separabilidade visual em `eda_pca_2d_top12.png`.
