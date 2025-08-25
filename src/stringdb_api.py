#!/usr/bin/env python3
"""
This script takes a list of genes (one per line in a text file),
fetches internal and external protein-protein interactions from STRING,
filters potential partner proteins based on degree and connection.

Requires:
    requests (install with `python -m pip install requests`)
"""

import pandas as pd
import requests


class StringDBAPI:
    """
    API client for retrieving protein-protein interactions from STRING DB.
    """

    # def __init__(self, config):
    #     self.species = config.get("species", 9606)
    #     self.score_threschold_string = config.get("score_threshold_stringdb"
    #     , 0.7)
    #     self.add_nodes = config.get("add_nodes", 30)

    def __init__(self, score, nodes):
        """
        Initialize the StringDBAPI instance.

        Args:
            score (float): Minimum interaction score threshold.
            nodes (int): Number of additional nodes to retrieve.
        """
        self.species = 9606
        self.score = score
        self.add_nodes = nodes

    def get_interactions(self, gene_list):
        """
        Get interactions from STRING for a given list of genes.

        Args:
            genes_list (list[str]): List of gene identifiers.

        Returns:
            list[dict]: JSON data from STRING DB API.
        """
        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": "%0A".join(gene_list),
            "species": self.species,
            "required_score": int(self.score * 1000),
            "add_nodes": self.add_nodes,
        }

        try:
            response = requests.get(url, params=params, timeout=100)
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
        Convert STRING JSON output into a cleaned pandas DataFrame.

        Args:
            json_data (list[dict]): JSON data from STRING DB API.

        Returns:
            pandas.DataFrame: Cleaned and filtered interaction table.
        """
        if not json_data:
            return pd.DataFrame()

        df = pd.DataFrame(json_data)

        if df.empty:
            return df

        df_clean = df.rename(
            columns={
                "preferredName_A": "gene_A",
                "preferredName_B": "gene_B",
                "score": "combined_score",
                "escore": "experimental_score",
                "dscore": "database_score",
            }
        )

        df_clean["source_db"] = "stringDB"
        df_clean["known_score"] = (
            df_clean["experimental_score"] + df_clean["database_score"]
        ) / 2

        essential_columns = ["gene_A", "gene_B", "known_score", "source_db"]
        df_final = df_clean[essential_columns].copy()

        df_filtered = df_final[df_final["known_score"] > self.score].copy()
        df_sorted = df_filtered.sort_values(by="known_score",
                                            ascending=False)

        return df_sorted
