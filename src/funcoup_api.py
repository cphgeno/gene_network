#!/usr/bin/env python3
"""
This script takes a list of genes (one per line in a text file),
fetches internal and external protein-protein interactions from FunCoup,
filters potential partner proteins based on degree and connection.

Requires:
    requests (install with `python -m pip install requests`)
    pandas
"""

import pandas as pd
import requests


class FunCoupAPI:
    """
    API client for retrieving protein-protein interactions from FunCoup DB.
    """

    def __init__(self, score, nodes, algo):
        """
        Initialize the FunCoupAPI instance.

        Args:
            score (float): Minimum interaction score threshold.
            nodes (int): Number of nodes per step to retrieve.
            algo (str): Expansion algorithm.
        """
        self.species = "9606"
        self.score_threshold_funcoup = score
        self.depth = 1
        self.node_per_step = nodes
        self.expansion_algorithm = algo
        self.prioritize_neighbours = "off"
        self.individual_evidence_only = "off"
        self.orthologs_only = "off"
        self.direction_threshold = 0

    def get_interactions(self, gene_list):
        """
        Get interactions from FunCoup for a given list of genes.

        Args:
            gene_list (list[str]): List of gene identifiers.
        Returns:
            dict: JSON data from FunCoup DB API.
        """
        base_url = "https://funcoup.org/api/json/network/"
        genes_str = ",".join(gene_list)
        url = (
            f"{base_url}{genes_str}&{self.species}"
            f"&{self.score_threshold_funcoup}&{self.direction_threshold}"
            f"&{self.depth}&{self.node_per_step}&{self.expansion_algorithm}"
            f"&{self.prioritize_neighbours}&''&{self.individual_evidence_only}"
            f"&{self.orthologs_only}&/"
        )

        try:
            response = requests.get(url, timeout=100)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            print(f"Request to {url} timed out. Please try again later.")
            return []

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []

    def json_pandas(self, json_data):
        """
        Convert FunCoup JSON output into a cleaned pandas DataFrame.

        Args:
            json_data (dict): JSON data from FunCoup DB API.
        Returns:
            pandas.DataFrame: Cleaned and filtered interaction table.
        """
        if not json_data:
            return pd.DataFrame()

        if not hasattr(json_data, "empty"):
            nodes = json_data["nodes"]
            links = json_data["links"]

            id_to_uniprot = {
                node_id: info["mappings"]["Gene_Symbol"]
                for node_id, info in nodes.items()
            }

            rows = []
            for key, val in links.items():
                src_id, tgt_id = key.split("|")
                rows.append(
                    {
                        "gene_A": id_to_uniprot.get(src_id),
                        "gene_B": id_to_uniprot.get(tgt_id),
                        "score_type": val["scoresPerGoldStandard"]["type"],
                        "confidence_ppv": float(
                            val["scoresPerGoldStandard"].get("ppv", 0)
                        ),
                        "known": val["scoresPerGoldStandard"]["known"],
                    }
                )

            df_links = pd.DataFrame(rows).copy()
            df_links["source_db"] = f"funcoup {self.expansion_algorithm}"

            df_filtered = df_links[df_links["known"] == "This"].copy()
            df_sorted = df_filtered.sort_values(by="score_type",
                                                ascending=False).copy()
            return df_sorted

        return pd.DataFrame()
