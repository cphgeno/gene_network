"""
Gene network analysis and optimization script.

This script loads a list of genes and a configuration file, then uses Optuna
to optimize the selection of gene interactions. The optimized network is
built, visualized, and evaluated using various metrics, and the results are
saved to output files.

Modules:
    - metrics: Provides functions for preparing gene data, building networks,
      visualizing graphs, and calculating metrics.
    - optuna_sampler: Contains the OptunaSampler class for running optimization
      studies on gene networks.
Usage:
    python append_genes.py config.json
Args:
    config.json: Path to a JSON configuration file containing:
        - input_file (str): Path to the input gene list file.
        - output_file (str): Path to the output Excel file for metrics.
        - output_graph (str): Path to save the network graph visualization.
        - trials_number (int): Number of Optuna trials to run.
        - repeats_number (int): Number of repetitions for each trial.
Outputs:
    - Excel file with gene-related metrics.
    - Image file of the network graph.
"""

import json
import sys
import time
from datetime import timedelta

from src import metrics
from src.optuna_sampler import OptunaSampler


def load_config_json(config_file):
    """
    Load configuration data from a JSON file.
    Args: config_file (str or Path): Path to the JSON configuration file.
    Returns: dict: Parsed configuration data as a Python dictionary.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)


def read_gene_file(filename):
    """
    Read a text file containing gene identifiers, one per line.
    Args: filename (str or Path): Path to the text file containing the gene
          list.
    Returns: list[str]: List of gene identifiers as strings, with whitespace
                        removed.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        UnicodeDecodeError: If the file cannot be decoded with UTF-8 encoding.
    """
    genes = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            genes.append(line.strip())
    return genes


def main():
    """
    Main entry point of the script.
    Loads configuration from a JSON file, reads the input gene list, and runs
    the Optuna optimization process to find robust gene interaction networks.
    The resulting network is visualized, gene metrics are calculated, and
    results are saved to output files.
    Workflow:
        1. Parse command-line arguments to get the config file path.
        2. Load configuration settings from JSON.
        3. Read the list of genes from the input file.
        4. Run Optuna optimization to select optimal interactions.
        5. Prepare data and build the gene interaction network.
        6. Visualize and save the network graph.
        7. Calculate and export gene-related metrics.
    Raises:
        SystemExit: If the required config file argument is missing.
        FileNotFoundError: If input files specified in the config do not exist.
    """
    if len(sys.argv) != 2:
        print("Usage python append_genes.py config.json")
        sys.exit(1)

    config = load_config_json(sys.argv[1])
    input_file = config["input_file"]
    print("Analysis of ", input_file)
    output_file = config["output_file"]
    output_graph = config["output_graph"]
    nb_trials = config["trials_number"]
    nb_repeats = config["run_number"]
    genes_list = read_gene_file(input_file)
    print("Genes list:" + str(sorted(genes_list)))
    start_time = time.time()

    optuna_sampler = OptunaSampler(genes_list, nb_repeats, nb_trials)
    interactions = optuna_sampler.run_optuna()
    robust_results = metrics.prepare_gene_data(interactions, genes_list)

    graph = metrics.build_network(robust_results)
    metrics.visualize_graph(graph, save_path=output_graph)

    info_added_genes = metrics.calculate_gene_metrics(graph, robust_results)
    info_added_genes.to_excel(output_file, index=False)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Execution time: {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()
