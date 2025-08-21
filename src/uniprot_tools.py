"""
uniprot_tools.py
================

This module provides functions to query UniProtKB for human gene annotations.
It allows retrieving UniProt IDs for specific genes and downloading associated
annotations in TSV format, with built-in retry handling for network
reliability.

Functions
---------
create_retry_session(retries=3, backoff_factor=0.3,
                     status_forcelist=(500, 502, 504))
    Create a requests.Session with automatic retries for transient HTTP errors.

get_uniprot_gene(gene)
    Retrieve the primary UniProt accession for a given human gene.

get_uniprot_tsv(uniprot_id)
    Download UniProt annotations in TSV format for the specified accession.

build_uniprot_dataframe(gene_list)
    Build a combined UniProt annotation DataFrame from a list of gene symbols.
"""

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_retry_session(
    retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)
):
    """
    Create a requests Session with automatic retries for failed requests.
    """
    current_session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,  # delay progression between retries
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    current_session.mount("http://", adapter)
    current_session.mount("https://", adapter)
    return current_session


session = create_retry_session()


def get_uniprot_gene(gene):
    """
    Retrieve the UniProt primary accession for a given human gene.

    Args
    gene : str
        Gene symbol to search in UniProt.

    Returns
    str or None
        UniProt primary accession if found, else None.
    """
    url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene_exact:{gene}+AND+organism_id:9606"
        f"&fields=gene_names"
    )

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to retrieve UniProt ID for gene {gene}: {e}")
        return None

    return next(
        (
            entry["primaryAccession"]
            for entry in data.get("results", [])
            if entry.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"
        ),
        None,
    )


def get_uniprot_tsv(uniprot_id):
    """
    Retrieve UniProt data in TSV format for a given accession.

    Args
    uniprot_id : str
        UniProt primary accession.

    Returns
    pandas.DataFrame or None
        DataFrame containing UniProt annotations, or None if retrieval fails.
    """
    url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        f"query=accession:{uniprot_id}+AND+organism_id:9606"
        "&format=tsv"
        "&fields=id,gene_names,gene_synonym,cc_activity_regulation,"
        "cc_pathway,cc_interaction,cc_subunit,cc_tissue_specificity,"
        "go_p,go_c,go_f,cc_disease,cc_pharmaceutical,cc_ptm,ft_signal"
    )

    try:
        return pd.read_csv(url, sep="\t")
    except (pd.errors.ParserError, OSError) as err:
        print(f"[ERROR] Failed to retrieve TSV data for {uniprot_id}: {err}")
        return None


def build_uniprot_dataframe(gene_list):
    """
    Build a UniProt annotation DataFrame from a list of genes.

    Args
    gene_list : list of str
        List of gene symbols.

    Returns
    pandas.DataFrame
        Combined DataFrame of UniProt annotations.
    """
    records = []

    for gene in gene_list:
        uniprot_id = get_uniprot_gene(gene)
        if not uniprot_id:
            print(f"[WARNING] No UniProt ID for gene {gene}")
            continue

        df = get_uniprot_tsv(uniprot_id)
        if df is None or df.empty:
            print(f"[WARNING] No data for UniProt ID {uniprot_id}")
            continue

        df["queried_gene"] = gene
        records.append(df)

    if records:
        return pd.concat(records, ignore_index=True)

    print("[INFO] No UniProt data retrieved.")
    return pd.DataFrame()
