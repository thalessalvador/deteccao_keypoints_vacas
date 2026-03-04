import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from ..util.io_arquivos import garantir_diretorio


def _colunas_feature(df: pd.DataFrame) -> List[str]:
    """
    Retorna apenas as colunas numericas de features, removendo metadados operacionais.

    Parametros:
        df (pd.DataFrame): Dataset de features completo.

    Retorno:
        List[str]: Lista de nomes de colunas tratadas como features.
    """
    colunas_excluir = {
        "arquivo",
        "classe",
        "target",
        "origem_instancia",
        "is_aug",
        "aug_id",
        "split_instancia",
    }
    candidatas = [c for c in df.columns if c not in colunas_excluir]
    numericas = [c for c in candidatas if pd.api.types.is_numeric_dtype(df[c])]
    return numericas


def _salvar_fig(fig: plt.Figure, caminho: Path) -> None:
    """
    Salva uma figura com layout ajustado e fecha o objeto.

    Parametros:
        fig (plt.Figure): Figura matplotlib.
        caminho (Path): Caminho do arquivo de saida.

    Retorno:
        None: Figura persistida em disco.
    """
    fig.tight_layout()
    fig.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analisar_features_exploratorio(config: Dict[str, Any], logger: logging.Logger) -> Path:
    """
    Executa analise exploratoria do dataset de features fora do pipeline principal.

    A analise gera artefatos em ``saidas/analise_features``:
    - resumo tabular em JSON/CSV;
    - graficos de distribuicao, correlacao, separabilidade e importancia;
    - relatorio markdown com insights automaticos.

    Parametros:
        config (Dict[str, Any]): Configuracao global carregada do YAML.
        logger (logging.Logger): Logger da aplicacao.

    Retorno:
        Path: Caminho do relatorio markdown gerado.
    """
    logger.info("=== Analise Exploratoria de Features ===")

    base_processado = Path(config["paths"]["processed"]) / "classificacao"
    caminho_csv = base_processado / "features" / "features_completas.csv"
    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo de features nao encontrado: {caminho_csv}. Rode 'gerar-features' antes."
        )

    dir_saida = Path(config["paths"]["outputs"]) / "analise_features"
    garantir_diretorio(dir_saida)

    df = pd.read_csv(caminho_csv)
    if df.empty:
        raise ValueError("features_completas.csv esta vazio. Nao ha dados para analisar.")

    cols_feat = _colunas_feature(df)
    if not cols_feat:
        raise ValueError("Nenhuma feature numerica encontrada para EDA.")

    # Higienizacao minima para EDA.
    X = df[cols_feat].replace([np.inf, -np.inf], np.nan)
    y = df["classe"].astype(str) if "classe" in df.columns else pd.Series(["sem_classe"] * len(df))

    resumo_geral = {
        "n_linhas": int(len(df)),
        "n_features": int(len(cols_feat)),
        "n_classes": int(y.nunique()),
        "classes_top10": y.value_counts().head(10).to_dict(),
        "tem_origem_instancia": bool("origem_instancia" in df.columns),
        "tem_split_instancia": bool("split_instancia" in df.columns),
    }

    if "origem_instancia" in df.columns:
        resumo_geral["origem_instancia"] = df["origem_instancia"].value_counts().to_dict()
    if "split_instancia" in df.columns:
        resumo_geral["split_instancia"] = df["split_instancia"].value_counts().to_dict()

    # 1) Distribuicao por classe.
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    vc = y.value_counts().sort_values(ascending=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax1, color="#3a7")
    ax1.set_title("Distribuicao de amostras por classe")
    ax1.set_xlabel("Classe")
    ax1.set_ylabel("Quantidade")
    ax1.tick_params(axis="x", rotation=90)
    _salvar_fig(fig1, dir_saida / "eda_distribuicao_classes.png")

    # 2) Missing por feature.
    missing_pct = X.isna().mean().sort_values(ascending=False) * 100.0
    missing_df = pd.DataFrame({"feature": missing_pct.index, "missing_pct": missing_pct.values})
    missing_df.to_csv(dir_saida / "eda_missing_por_feature.csv", index=False)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    top_missing = missing_df.head(25)
    sns.barplot(data=top_missing, x="feature", y="missing_pct", ax=ax2, color="#d97")
    ax2.set_title("Top features com maior percentual de missing")
    ax2.set_xlabel("Feature")
    ax2.set_ylabel("Missing (%)")
    ax2.tick_params(axis="x", rotation=90)
    _salvar_fig(fig2, dir_saida / "eda_missing_top25.png")

    # 3) Correlacao (apenas features com variancia > 0 e sem NaN para correlacao).
    X_corr = X.fillna(0.0)
    variancias = X_corr.var(numeric_only=True)
    cols_var = variancias[variancias > 0].index.tolist()
    corr = X_corr[cols_var].corr(numeric_only=True, method="pearson")
    corr.to_csv(dir_saida / "eda_correlacao.csv", index=True)

    fig3, ax3 = plt.subplots(figsize=(14.3, 11.7))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax3, cbar_kws={"shrink": 0.7})
    ax3.set_title("Matriz de correlacao das features")
    ax3.tick_params(axis="x", labelsize=8, rotation=90)
    ax3.tick_params(axis="y", labelsize=8, rotation=0)
    _salvar_fig(fig3, dir_saida / "eda_heatmap_correlacao.png")

    # 4) Importancia univariada (ANOVA F + Mutual Information).
    X_imp = X.fillna(0.0)
    var_imp = X_imp.var(numeric_only=True)
    features_constantes = var_imp[var_imp <= 0.0].index.tolist()
    if features_constantes:
        logger.warning(
            "EDA: %d features constantes foram removidas da analise de importancia: %s",
            len(features_constantes),
            features_constantes,
        )
        pd.DataFrame({"feature_constante": features_constantes}).to_csv(
            dir_saida / "eda_features_constantes.csv",
            index=False,
        )
    cols_imp = [c for c in cols_feat if c not in set(features_constantes)]
    if not cols_imp:
        raise ValueError("Nao ha features variaveis para calcular importancia.")
    X_imp = X_imp[cols_imp]
    y_codes = pd.Categorical(y).codes
    f_vals, _ = f_classif(X_imp, y_codes)
    mi_vals = mutual_info_classif(X_imp, y_codes, random_state=42)
    imp_df = pd.DataFrame(
        {
            "feature": cols_imp,
            "anova_f": np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0),
            "mutual_info": np.nan_to_num(mi_vals, nan=0.0, posinf=0.0, neginf=0.0),
        }
    )
    imp_df["rank_media"] = (
        imp_df["anova_f"].rank(ascending=False, method="average")
        + imp_df["mutual_info"].rank(ascending=False, method="average")
    ) / 2.0
    imp_df = imp_df.sort_values(
        ["rank_media", "mutual_info", "anova_f"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    imp_df["posicao_importancia"] = np.arange(1, len(imp_df) + 1)
    imp_df.to_csv(dir_saida / "eda_importancia_features.csv", index=False)

    top_imp = imp_df.head(25).sort_values("rank_media", ascending=False)
    fig4, ax4 = plt.subplots(figsize=(11, 10))
    sns.barplot(data=top_imp, x="rank_media", y="feature", ax=ax4, color="#58a")
    ax4.set_title("Top-25 features mais informativas (ranking medio)")
    ax4.set_xlabel("Ranking medio (menor = melhor)")
    ax4.set_ylabel("Feature")
    _salvar_fig(fig4, dir_saida / "eda_top25_importancia.png")

    # 5) Separabilidade em PCA 2D.
    X_pca = X.fillna(0.0).astype(float)
    X_pca = StandardScaler().fit_transform(X_pca)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_pca)
    pca_df = pd.DataFrame({"pc1": proj[:, 0], "pc2": proj[:, 1], "classe": y.values})
    pca_df.to_csv(dir_saida / "eda_pca_2d.csv", index=False)

    classes_top = y.value_counts().head(12).index.tolist()
    pca_plot = pca_df[pca_df["classe"].isin(classes_top)].copy()
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=pca_plot,
        x="pc1",
        y="pc2",
        hue="classe",
        s=22,
        alpha=0.75,
        ax=ax5,
    )
    ax5.set_title("PCA 2D (top-12 classes por volume)")
    ax5.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax5.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax5.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    _salvar_fig(fig5, dir_saida / "eda_pca_2d_top12.png")

    # 6) Sumario JSON e markdown.
    pares_corr = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "corr"})
    )
    pares_corr["corr_abs"] = pares_corr["corr"].abs()
    top_corr = pares_corr.sort_values("corr_abs", ascending=False).head(10)
    top_corr.to_csv(dir_saida / "eda_top_correlacoes.csv", index=False)

    resumo = {
        "gerado_em": pd.Timestamp.now().isoformat(),
        "arquivo_origem": str(caminho_csv),
        "resumo_geral": resumo_geral,
        "pca_var_exp": {
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]),
        },
        "top10_correlacoes_abs": top_corr.to_dict(orient="records"),
        "top25_features_rank": imp_df.head(25).to_dict(orient="records"),
    }
    with open(dir_saida / "resumo_eda.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    md_path = dir_saida / "relatorio_eda.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Relatorio EDA de Features\n\n")
        f.write("## Resumo geral\n")
        f.write(f"- Amostras: **{resumo_geral['n_linhas']}**\n")
        f.write(f"- Features numericas: **{resumo_geral['n_features']}**\n")
        f.write(f"- Classes: **{resumo_geral['n_classes']}**\n")
        if "origem_instancia" in resumo_geral:
            f.write(f"- Origem das instancias: **{resumo_geral['origem_instancia']}**\n")
        if "split_instancia" in resumo_geral:
            f.write(f"- Split das instancias: **{resumo_geral['split_instancia']}**\n")
        f.write("\n")
        f.write("## Artefatos gerados\n")
        f.write("- `eda_distribuicao_classes.png`\n")
        f.write("- `eda_missing_top25.png`\n")
        f.write("- `eda_heatmap_correlacao.png`\n")
        f.write("- `eda_top25_importancia.png`\n")
        f.write("- `eda_pca_2d_top12.png`\n")
        f.write("- `eda_missing_por_feature.csv`\n")
        f.write("- `eda_correlacao.csv`\n")
        f.write("- `eda_importancia_features.csv`\n")
        f.write("- `eda_top_correlacoes.csv`\n")
        f.write("- `resumo_eda.json`\n\n")
        if features_constantes:
            f.write("- `eda_features_constantes.csv`\n\n")
            f.write("Features constantes removidas da analise de importancia:\n")
            for nome in features_constantes:
                f.write(f"- `{nome}`\n")
            f.write("\n")
        f.write("## Principais achados automaticos\n")
        f.write("### Top-25 features por ranking medio (ANOVA + MI)\n")
        for _, row in imp_df.head(25).iterrows():
            f.write(
                f"- `{row['feature']}` | rank={row['rank_media']:.2f} | "
                f"anova_f={row['anova_f']:.4f} | mi={row['mutual_info']:.4f}\n"
            )
        f.write("\n")
        f.write("### Top correlacoes absolutas\n")
        for _, row in top_corr.iterrows():
            f.write(f"- `{row['feature_a']}` x `{row['feature_b']}`: corr={row['corr']:.4f}\n")
        f.write("\n")
        f.write(
            "### Observacao\n"
            "Este relatorio e descritivo e nao substitui a avaliacao final do classificador.\n"
            "Use os resultados para orientar selecao de features, reducao de redundancia e novos experimentos.\n"
        )

    logger.info(f"EDA concluida. Relatorio: {md_path}")
    return md_path
