# Gene Network
**Author:** Alice Hermann

A short description: this project extends a gene list using StringDB and FunCoup. Optuna, an optimization framework, is used to automatically 
tune parameters for network expansion, filtering, and robustness assessment. The optimization is based on F1-score.
It requires a configuration file (JSON) specifying:
- the input file with a gene list (one gene per line)
- the number of trials and the number of runs per trial for Optuna
- the name of the output file (XLSX)
- the name of the output graph (PNG)

The added genes in the network are saved in a file with information from UniProt.

---
## Running the project

Run the main script with your configuration file:
```bash
python main.py config.json
```

---
## Example of configuration file

```json

{
"input_file": "datas/genes.txt",
"trials_number": 100,
"runs_number": 10,
"output_file": "results/output.xlsx",
"output_graph": "results/network.png"
}

```

---

## Example of output file

In the excel file: 
- gene name
- degree centrality for the gene
- betweenness centrality for the gene
- closeness centrality
- eigenvector centrality
- degree of the gene
- number of direct neighbours
- interaction type
- number of direct connections with genes from original list
- neighbours from original list
- source number for this interaction
- sources for these interactions
- information via UniProt (activity regulation, pathway, structure, tissue specificity, ...)

The PNG is the visualization of the gene network:
- **Red nodes:** original genes from the gene list
- **White nodes:** added genes

---
## Requirements
This project requires the following Python packages:
- Optuna
- requests
- NetworkX
- NumPy
- pandas
- scikit-learn

Install via:
```bash
pip install -r requirements.txt
```

---
## Project Structure

- `src/` : source code
- `datas/` : gene lists
- `results/` : results files
- `config/` : configuration files
---

## Notes
- The configuration file is mandatory for running `main.py`.
- The output includes both an XLSX file and a PNG graph of the network.

EOF
