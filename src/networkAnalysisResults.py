"""
This module defines the NetworkAnalysisResults class, which stores and analyzes
gene interaction network iterations. It allows adding per-iteration data and
computing aggregated edge statistics (e.g., frequency, percentage, average
score, and metadata).
"""

from collections import defaultdict

import pandas as pd


class NetworkAnalysisResults:
    """Store per-iteration gene network results and compute edge statistics."""

    def __init__(self, original_genes):
        """
        Initialize the results container.

        Args
        original_genes : list[str] or set[str] or None
            The reference set of original genes. If None, only added genes
            will be tracked.
        """
        self.iterations = []  # Store each iteration's data
        self.metrics = {}  # Calculated metrics storage
        self.original_genes = (
            set(original_genes) if original_genes is not None else None
        )

    def add_iteration(
        self,
        links_df,
        genes_original_present,
        genes_added,
        all_genes,
        genes_original_missing,
        graph_nx,
        iteration_id,
    ):
        """
        Add data for one iteration.

        Args
        links_df : pd.DataFrame
            DataFrame with at least columns ['gene_A', 'gene_B'] representing
            edges.
        genes_original_present : set or list
            Original genes present in this iteration.
        genes_added : set or list
            Genes added in this iteration.
        all_genes : set or list
            All genes in this iteration.
        genes_original_missing : set or list
            Original genes missing in this iteration.
        graph_nx : networkx.Graph
            Graph object representing this iteration.
        iteration_id : int or str
            Identifier of the iteration.
        """
        self.iterations.append(
            {
                "iteration_id": iteration_id,
                "links_df": links_df,
                "genes_original_present": set(genes_original_present),
                "genes_added": set(genes_added),
                "all_genes": set(all_genes),
                "genes_original_missing": set(genes_original_missing),
                "graph_nx": graph_nx,
            }
        )

    def compute_edge_stats_with_metadata(self, nb_iterations):
        """
        Aggregate edge statistics across all iterations.

        Each edge is defined as an undirected pair (gene_A, gene_B), sorted
        alphabetically. For each edge, the method computes:
        - number of occurrences across iterations,
        - percentage of presence,
        - average score,
        - link type (original-original, added-added, or original-added),
        - metadata from the best scoring occurrence.
        Args
        nb_iterations : int
            Number of iterations to consider (used for normalization).
        Returns
        pd.DataFrame
            Aggregated edge statistics with metadata.
        """
        edge_counter = defaultdict(int)
        score_sums = defaultdict(float)
        best_rows = {}

        total = len(self.iterations)

        for it in self.iterations:
            df = it["links_df"].copy()

            for gene_col in ["gene_A", "gene_B"]:
                if f"{gene_col}_type" not in df.columns:
                    raise ValueError(f"Missing column '{gene_col}_type'"
                                     f"in links_df")

            for _, row in df.iterrows():
                gene1, gene2 = row["gene_A"], row["gene_B"]
                type1, type2 = row["gene_A_type"], row["gene_B_type"]
                score = row["score"]

                # Alphabetically sort gene pair and align gene types
                if gene1 < gene2:
                    edge = (gene1, gene2)
                    type_pair = (type1, type2)
                else:
                    edge = (gene2, gene1)
                    type_pair = (type2, type1)

                edge_counter[edge] += 1
                score_sums[edge] += score

                # Keep row with the best score for this edge
                if edge not in best_rows or score > best_rows[edge]["score"]:
                    new_row = row.copy()
                    new_row["gene_A"] = edge[0]
                    new_row["gene_B"] = edge[1]
                    new_row["gene_A_type"] = type_pair[0]
                    new_row["gene_B_type"] = type_pair[1]
                    best_rows[edge] = new_row

        records = []
        for edge, row in best_rows.items():
            count = edge_counter[edge]
            mean_score = score_sums[edge] / count
            type_a, type_b = row["gene_A_type"], row["gene_B_type"]

            # Define link type
            if type_a == type_b:
                link_type = f"{type_a}-{type_b}"
            else:
                link_type = "original-added"

            record = row.to_dict()
            record.update(
                {
                    "gene_A": edge[0],
                    "gene_B": edge[1],
                    "nb_occurrences": count,
                    "pourcentage": count / total,
                    "score_moyen": mean_score,
                    "link_type": link_type,
                }
            )
            records.append(record)

        return pd.DataFrame(records).reset_index(drop=True)
