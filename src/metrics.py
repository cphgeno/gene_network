"""
Gene Network Construction and Analysis
======================================

This module provides utilities to process, integrate, and analyze gene
interaction data from multiple sources (e.g., STRING, FunCoup).
It follows a multi-step workflow:

1. **Prepare Data** (`prepare_gene_data`):
   - Merge multiple interaction DataFrames.
   - Standardize column names (`gene_A`, `gene_B`, `score`).
   - Identify original, added, and missing genes.
   - Deduplicate interactions by keeping only the highest-scoring pair.

2. **Build Network** (`build_network`):
   - Construct a weighted undirected `networkx.Graph`.
   - Each edge stores composite weights, number of sources, interaction types,
     and the original scores.
   - Nodes are annotated as `original` or `added` with their degree.

3. **Visualize Graph** (`visualize_graph`):
   - Draws the network with color-coded edges based on data sources
     (STRING, FunCoup).
   - Nodes are colored by type: red (original) vs white (added).
   - Node size is proportional to degree centrality.

4. **Compute Gene Metrics** (`calculate_gene_metrics`):
   - Calculates multiple centrality measures (degree, betweenness,
     closeness, eigenvector).
   - Quantifies direct connections to original genes and source presence.
   - Annotates added genes with UniProt information via `uniprot_tools`.

Dependencies
------------
- `pandas`, `numpy`
- `networkx`
- `matplotlib`
- local module `.uniprot_tools` for UniProt annotations

Notes
-----
- The workflow assumes interactions include `source_db` and a confidence score
  (`known_score` for STRING or `confidence_ppv` for FunCoup).
- If all inputs are empty, fallback objects are returned (empty DataFrame,
  empty sets, empty graph).
- Visualization is optional: graphs can be displayed interactively or saved
  to a file.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import uniprot_tools as ag


def prepare_gene_data(df_list, genes_list):
    """
    Prepare and clean gene interaction data from multiple sources
    (STRING, FunCoup).
    This function:
      1. Merges one or several DataFrames of gene interactions.
      2. Ensures required columns (`gene_A`, `gene_B`, `score`) are present,
         even if the input DataFrames are empty or incomplete.
      3. Builds sets of all genes, original genes, added genes, and missing
         genes.
      4. Classifies each gene as 'original' (in the input gene list) or 'added'
         (introduced by STRING/FunCoup).
      5. Removes duplicate interactions by keeping only the highest-score
         interaction per gene pair.
    Args
    df_list : list[pd.DataFrame] or pd.DataFrame
        List of interaction DataFrames (e.g. from STRING and FunCoup APIs).
        Each DataFrame must contain at least 'gene_A' and 'gene_B'.
        Can also be a single DataFrame or an empty list.
    genes_list : list[str]
        List of original genes provided by the user (reference gene set).
    Returns
    dict
        Dictionary with the following keys:
        - 'combined_df' : pd.DataFrame
            Cleaned and merged interactions with standardized columns.
        - 'all_genes' : set
            Set of all genes present in the merged network.
        - 'original_genes' : set
            Subset of `genes_list` found in the interactions.
        - 'added_genes' : set
            Genes not in the original list but introduced by STRING/FunCoup.
        - 'genes_classification' : dict
            Mapping gene → "original" or "added".
        - 'missing_original_genes' : set
            Genes from `genes_list` not found in the merged interactions.
    Notes
    -----
    - If all input DataFrames are empty or None, the function will return an
      empty DataFrame with standard columns and sets will default to empty.
    - The `score` column is built from `known_score` or `confidence_ppv` if
      available; otherwise it is filled with NaN.
    """
    if isinstance(df_list, pd.DataFrame):
        combined_df = df_list.copy()
    elif isinstance(df_list, list) and len(df_list) > 0:
        dfs = [df for df in df_list if df is not None and not df.empty]
        if len(dfs) == 0:
            combined_df = pd.DataFrame(columns=["gene_A", "gene_B", "score"])
        else:
            combined_df = pd.concat(dfs, ignore_index=True).copy()
    else:
        combined_df = pd.DataFrame(columns=["gene_A", "gene_B", "score"])

    for col in ["gene_A", "gene_B"]:
        if col not in combined_df.columns:
            combined_df[col] = []

    if "score" not in combined_df.columns:
        has_known = "known_score" in combined_df.columns
        has_conf = "confidence_ppv" in combined_df.columns

        if has_known or has_conf:
            combined_df["score"] = combined_df.get(
                "known_score", pd.Series(dtype=float)
            ).fillna(combined_df.get("confidence_ppv", pd.Series(dtype=float)))
        else:
            combined_df["score"] = pd.Series(dtype=float)

    if combined_df.empty:
        return {
            "combined_df": combined_df,
            "all_genes": set(),
            "original_genes": set(),
            "added_genes": set(),
            "genes_classification": {},
            "missing_original_genes": set(genes_list),
        }

    original_genes_set = set(genes_list)
    gene_classification = {}

    all_genes = set(combined_df["gene_A"].unique()) | set(
        combined_df["gene_B"].unique()
    )
    original_genes = all_genes & original_genes_set
    added_genes = all_genes - original_genes_set
    missing_original = original_genes_set - all_genes

    for gene in all_genes:
        if gene in original_genes_set:
            gene_classification[gene] = "original"
        else:
            gene_classification[gene] = "added"

    combined_df["gene_A_type"] = combined_df["gene_A"].map(gene_classification)
    combined_df["gene_B_type"] = combined_df["gene_B"].map(gene_classification)

    combined_df["pair"] = combined_df.apply(
        lambda row: tuple(sorted([row["gene_A"], row["gene_B"]])), axis=1
    )
    combined_df = combined_df.sort_values("score", ascending=False)
    combined_df = combined_df.drop_duplicates(subset="pair", keep="first")
    combined_df = combined_df.drop(columns="pair")

    return {
        "combined_df": combined_df.reset_index(drop=True),
        "all_genes": all_genes,
        "original_genes": original_genes,
        "added_genes": added_genes,
        "genes_classification": gene_classification,
        "missing_original_genes": missing_original,
    }


def build_network(results):
    """
    Build a NetworkX graph from the combined gene interaction data.
    Each node is a gene, with attributes:
      - 'type': 'original' or 'added'.
      - 'degree': number of connections.

    Each edge has attributes:
      - 'weight': average score of all interactions between the two genes.
      - 'nb_sources': number of different sources reporting the interaction.
      - 'sources': list of sources.
      - 'interaction_type': list of interaction types.
      - 'individual_scores': list of all scores for this pair.

    Args : results (dict) : Output from `prepare_gene_data()`.
    Returns: nx.Graph: NetworkX graph representing the gene interactions.
    """
    combined_df = results["combined_df"]

    global_graph = nx.Graph()
    edge_weights = defaultdict(list)
    edge_sources = defaultdict(list)
    edge_type = defaultdict(list)

    for _, row in combined_df.iterrows():
        gene_A, gene_B = row["gene_A"], row["gene_B"]
        source = row["source_db"]

        if source == "stringDB":
            score = row["known_score"]
            interaction_type = "PPI"
        else:
            score = row["confidence_ppv"]
            interaction_type = row["score_type"]

        edge_key = tuple(sorted([gene_A, gene_B]))
        edge_weights[edge_key].append(score)
        edge_sources[edge_key].append(source)
        edge_type[edge_key].append(interaction_type)

    for edge, score_list in edge_weights.items():
        gene_a, gene_b = edge
        weight_composite = np.mean(score_list)
        nb_sources = len(edge_sources[edge])

        global_graph.add_edge(
            gene_a,
            gene_b,
            weight=weight_composite,
            nb_sources=nb_sources,
            sources=edge_sources[edge],
            interaction_type=edge_type[edge],
            individual_scores=score_list,
        )

        node_types = (
            pd.concat(
                [
                    combined_df[["gene_A", "gene_A_type"]].rename(
                        columns={"gene_A": "gene", "gene_A_type": "type"}
                    ),
                    combined_df[["gene_B", "gene_B_type"]].rename(
                        columns={"gene_B": "gene", "gene_B_type": "type"}
                    ),
                ]
            )
            .drop_duplicates()
            .set_index("gene")["type"]
            .to_dict()
        )

        for node in global_graph.nodes():
            global_graph.nodes[node]["type"] = node_types.get(node, "unknown")
            global_graph.nodes[node]["degree"] = global_graph.degree[node]

    return global_graph


def visualize_graph(global_graph, figsize=(15, 22), save_path=None):
    """
    Visualize the gene interaction network with color-coded nodes and edges.
    Nodes:
      - Red: original genes.
      - White: added genes.
      - Node size proportional to degree.
    Edges:
      - Color-coded by source database ('stringDB', 'funcoup group',
                                                    'funcoup maxlink').

    Parameters
    ----------
    global_graph : nx.Graph: Graph generated by `build_network()`.
    figsize : tuple, optional: Figure size (width, height) in inches.
    save_path : str, optional
        If provided, saves the figure to this path. Otherwise, shows it.
    """
    source_color_map = {
        "stringDB": "blue",
        "funcoup group": "green",
        "funcoup maxlink": "orange",
    }

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(global_graph, k=1, iterations=50)

    original_nodes = [
        n for n in global_graph.nodes() if global_graph.nodes[n]["type"] == "original"
    ]
    added_nodes = [
        n for n in global_graph.nodes() if global_graph.nodes[n]["type"] == "added"
    ]
    edge_colors = []
    for _, _, data in global_graph.edges(data=True):
        sources = data.get("sources", [])
        primary_source = sources[0] if sources else "unknown"
        edge_colors.append(source_color_map.get(primary_source, "gray"))

    pos = nx.spring_layout(global_graph, k=1, iterations=50, seed=42)
    degrees = dict(global_graph.degree())
    max_degree = max(degrees.values())

    node_sizes_original = [
        300 + (degrees[node] / max_degree) * 500 for node in original_nodes
    ]
    node_sizes_added = [
        300 + (degrees[node] / max_degree) * 500 for node in added_nodes
    ]

    nx.draw_networkx_edges(
        global_graph, pos, alpha=0.3, width=0.5, edge_color=edge_colors
    )

    if original_nodes:
        nx.draw_networkx_nodes(
            global_graph,
            pos,
            nodelist=original_nodes,
            node_color="red",
            node_size=node_sizes_original,
            alpha=0.8,
            edgecolors="lightyellow",
        )
    if added_nodes:
        nx.draw_networkx_nodes(
            global_graph,
            pos,
            nodelist=added_nodes,
            node_color="white",
            node_size=node_sizes_added,
            alpha=0.8,
            edgecolors="yellow",
        )

    nx.draw_networkx_labels(global_graph, pos, font_size=8, font_weight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def calculate_gene_metrics(global_graph, results):
    """
    Compute centrality metrics and connection info for each added gene.
    Metrics include:
      - degree, degree_centrality
      - betweenness_centrality
      - closeness_centrality
      - eigenvector_centrality
      - number of direct neighbors
      - types of interactions
      - connections to original genes
      - sources of interaction data

    Args
    global_graph: nx.Graph : NetworkX graph generated by `build_network()`.
    results: dict : Output from `prepare_gene_data()`.
    Returns
    pd.DataFrame: DataFrame with one row per added gene containing all metrics.
    """
    combined_df = results["combined_df"]
    added_genes = results["added_genes"]
    original_genes = results["original_genes"]

    degree_centrality = nx.degree_centrality(global_graph)
    betweenness_centrality = nx.betweenness_centrality(global_graph)
    closeness_centrality = nx.closeness_centrality(global_graph)

    try:
        eigenvector_centrality = nx.eigenvector_centrality(global_graph, max_iter=1000)
    except nx.NetworkXException as err:
        print(f"Error computing eigenvector_centrality: {err}")
        eigenvector_centrality = degree_centrality

    def connection_to_originals(gene):
        neighbors = set(global_graph.neighbors(gene))
        original_neighbors = neighbors & original_genes
        if len(original_genes) == 0:
            ratio = 0
        else:
            ratio = len(original_neighbors) / len(original_genes)

        return {
            "nb_direct_connections": len(original_neighbors),
            "ratio_connections_original": ratio,
            "original_neighbors": original_neighbors,
        }

    def get_source_presence(gene):
        """
        Determine in how many sources a given gene appears.
        
        Args
        gene : str: Gene name
        combined_df : pd.DataFrame: DataFrame of all interactions.
        Returns
        dict
            - 'nb_sources_total': number of distinct sources.
            - 'sources_list': list of source names.
        """
        sources = set()
        gene_interactions = combined_df[
            (combined_df["gene_A"] == gene) | (combined_df["gene_B"] == gene)
        ]
        sources = set(gene_interactions["source_db"].unique())

        return {"nb_sources_total": len(sources), "sources_list": list(sources)}

    metrics_list = []

    for i, gene in enumerate(added_genes):
        metrics = {"gene": gene}
        metrics["degree_centrality"] = degree_centrality.get(gene, 0)
        metrics["betweenness_centrality"] = betweenness_centrality.get(gene, 0)
        metrics["closeness_centrality"] = closeness_centrality.get(gene, 0)
        metrics["eigenvector_centrality"] = eigenvector_centrality.get(gene, 0)
        metrics["degree"] = global_graph.degree(gene)

        edge_types = []
        for neighbor in global_graph.neighbors(gene):
            edge_data = global_graph[gene][neighbor]
            edge_types.extend(edge_data.get("interaction_type", []))

        metrics["direct_neighbors"] = list(global_graph.neighbors(gene))
        metrics["interaction_types"] = list(set(edge_types))
        orig_connections = connection_to_originals(gene)
        metrics.update(orig_connections)

        source_presence = get_source_presence(gene)
        metrics.update(source_presence)

        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    print("Searching informations from UniProt")
    uniprot_df = ag.build_uniprot_dataframe(added_genes)

    df_merged = metrics_df.merge(
        uniprot_df, left_on="gene", right_on="queried_gene", how="left"
    )
    df_merged.drop(columns=["queried_gene"], inplace=True)

    added_genes_in_graph = [g for g in added_genes if g in global_graph]
    print(f"Added genes present in the graph: {len(added_genes_in_graph)}")

    if len(added_genes_in_graph) == 0:
        print("No added genes present in the graphe.")
    return df_merged
