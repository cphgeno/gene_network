"""
    Class for sampling gene network parameters using Optuna optimization.
    It evaluates combinations of score thresholds and number of added genes
    to maximize f1_score
"""
from sklearn.model_selection import train_test_split
import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner
from src import StringDBAPI
from src import FunCoupAPI
from src import metrics
from src import NetworkAnalysisResults


class OptunaSampler:
    """
    Initialize the sampler with a list of genes.
    Parameters:
    - genes_list (List[str]): Original list of genes to use as input.
    - n_repeats (int): repeats number of one trial
    - n_trials (int): trials number
    """

    def __init__(self, genes_list, n_repeats=10, n_trials=100):
        self.genes_list = genes_list
        self.n_repeats = n_repeats
        self.n_trials = n_trials
        self.folds_train = []
        self.folds_hidden = []

    def build_graph(self, iteration, results, gene_subset, confidence,
                    n_genes_to_add):
        """
        Build a gene interaction network by querying STRING and FunCoup APIs.

        Parameters:
        - iteration (int): ID of the current iteration (used for tracking).
        - results (NetworkAnalysisResults): Object that stores metrics for
                                            each iteration.
        - gene_subset (List[str]): Subset of genes to include in the network.
        - confidence (float): Minimum confidence score to keep an
                                            interaction.
        - n_genes_to_add (int): Number of external genes to add to the network.

        Returns:
        - pd.DataFrame: DataFrame with gene coverage statistics for this
                        iteration.
        """
        api_string = StringDBAPI(confidence, n_genes_to_add)
        api_funcoup = FunCoupAPI(confidence, n_genes_to_add, 'group')
        api_funcoup2 = FunCoupAPI(confidence, n_genes_to_add, 'maxlink')
        partners_string = api_string.get_interactions(gene_subset)
        result_string = api_string.json_pandas(partners_string)
        partners_funcoup = api_funcoup.get_interactions(gene_subset)
        result_funcoup_group = api_funcoup.json_pandas(partners_funcoup)
        partners_funcoup2 = api_funcoup2.get_interactions(gene_subset)
        result_funcoup_maxlink = api_funcoup2.json_pandas(partners_funcoup2)
        result = metrics.prepare_gene_data([result_string,
                                            result_funcoup_group,
                                            result_funcoup_maxlink],
                                           self.genes_list)
        graph = metrics.build_network(result)
        results.add_iteration(
                links_df=result['combined_df'],
                genes_original_present=result['original_genes'],
                genes_added=result['added_genes'],
                all_genes=result['all_genes'],
                genes_original_missing=result['missing_original_genes'],
                graph_nx=graph,
                iteration_id=iteration
            )

    def genes_set(self, df):
        """
        Calculate the dictionnary of genes from the interactions's DataFrame.

        Parameters:
        - dataFrame: dataFrame with the interactions
        Returns:
        - dictionnary: dictionnary with the set of original genes and
                        added genes in the network.
        """

        all_genes = set(df['gene_A'].unique()) | set(df['gene_B'].unique())
        original_genes = all_genes & set(self.genes_list)
        added_genes = all_genes - set(original_genes)

        gene_sets = {}
        gene_sets["original"] = sorted(original_genes)
        gene_sets["added"] = sorted(added_genes)
        gene_sets["all"] = sorted(all_genes)
        return gene_sets

    def calcultate_fold_f1(self, gene_set, gene_training, gene_hidden):
        """
        Calculate the score_f1 from the interactions's DataFrame.

        Parameters:
        - dictionnary: dictionnary with the set of original genes and
                        added genes in the network.

        Returns:
        - float: score_f1 value between 0 and 1.
        """
        if len(gene_set["added"]) == 0:
            return 0
        # Precision and Recall
        true_positive = len(set(gene_set['all']) & set(gene_hidden))
        added_genes = len(set(gene_set['all']) - set(gene_training))
        if added_genes > 0:
            precision = true_positive / added_genes
            recall = true_positive / len(set(gene_hidden))
        else:
            precision = 0
            recall = 0

        if (precision + recall) > 0:
            score_f1 = (2 * precision * recall) / (precision + recall)
        else:
            score_f1 = 0

        return score_f1

    def compute_macro_f1(self, f1_score_list):
        """
        Compute Macro-F1 across folds.

        Parameters:
        - list: list of f1_score from the folds

        Returns:
        - float: score_f1 value between 0 and 1.
        """
        return sum(f1_score_list) / len(f1_score_list)

    def analyse_best_trials(self, study, tol=1e-9):
        """
        Analyze all trials that achieved the best objective value in
        an Optuna study and check whether their `user_attrs["robust_genes"]`
        dictionaries are identical.

        Parameters
        ----------
        study : optuna.Study
            The Optuna study object after optimization.
        tol : float, optional (default=1e-9)
            Numerical tolerance for considering two trial values as equal.
            Useful when results differ only due to floating-point precision.

        Behavior
        --------
        - Finds all trials whose value is within `tol` of the study's
            best value.
        - Compares the `user_attrs["robust_genes"]` dictionary across
            these trials.
        - If all are identical, prints a confirmation message.
        - If differences are found, prints details for each best trial:
        - Trial number
        - Objective value
        - Hyperparameters (`params`)
        - `robust_genes` dictionary

        Notes
        -----
        This function is intended to be run *after* `study.optimize()` has
        finished.
        """
        best_value = study.best_value
        best_trials = [
            t for t in study.trials
            if t.value is not None and abs(t.value - best_value) <= tol
        ]

        if not best_trials:
            print("No trials found with the best value.")
            return

        try:
            reference_dict = best_trials[0].user_attrs["robust_genes"]
        except KeyError:
            print("⚠ Some trials do not contain the key 'robust_genes'.")
            return

        # Check if all robust_genes dicts are the same
        all_same = all(
            t.user_attrs.get("robust_genes") == reference_dict
            for t in best_trials
        )

        if all_same:
            print(f"All {len(best_trials)} best trials "
                  f"have identical best results.")
        else:
            print(f"Differences found in 'robust_genes'"
                  f"among {len(best_trials)} best trials:")
            for t in best_trials:
                print(f"\nTrial #{t.number}")
                print(f"  Value: {t.value}")
                print(f"  Params: {t.params}")
                print(f"  robust_genes: {t.user_attrs.get('robust_genes')}")

    def objective(self, trial):
        """
        Objective with pruning and intermediate reporting.

        Parameters:
        - trial (optuna.Trial): Trial object where parameters are suggested.

        Returns:
        - float: Ojective (f1 score) to maximize.
        """
        confidence = trial.suggest_float("confidence", 0.7, 0.9,
                                         step=0.01)
        n_genes_to_add = trial.suggest_int("n_genes_to_add", 5, 50)
        conservation_threshold = trial.suggest_float("conservation_threshold",
                                                     0.3, 0.95, step=0.01)
        filtered = []
        f1_scores = []
        score = 0
        i = 0
        seed = 42

        for i in range(self.n_repeats):
            train_genes = self.folds_train[i]
            hidden_genes = self.folds_hidden[i]
            results = NetworkAnalysisResults(train_genes)
            for j in range(10):
                seed = seed + j  # change seed per iterationx
                fold_genes, hidden_fold = train_test_split(train_genes,
                                                           test_size=0.2,
                                                           random_state=seed)
                self.build_graph(j, results, fold_genes,
                                 confidence,
                                 n_genes_to_add)
                results_total = results.compute_edge_stats_with_metadata(j+1)
            mask = results_total['pourcentage'] >= conservation_threshold
            filtered = results_total[mask].copy()
            genes_set = self.genes_set(filtered)
            f1_score = self.calcultate_fold_f1(genes_set,
                                               train_genes,
                                               hidden_genes
                                               )
            f1_scores.append(f1_score)

            score = self.compute_macro_f1(f1_scores)
            trial.report(score, step=i+1)
            # Vérifier si on prune
            if trial.should_prune():
                print(
                    f"Trial {trial.number} pruned at step {i+1} "
                    f"with value {score} and parameters {trial.params}"
                    )
                raise optuna.TrialPruned()
            i = i + 1

        trial.set_user_attr("graph_filtered", filtered)
        trial.set_user_attr("robust_genes", genes_set)

        return score

    def generate_fixed_train(self, seed=42):
        """
        Generate k-fold partitions by removing a subset of genes
        acccording to n_repeats.

        Parameters:
        - seed (int): Random seed for reproducibility.

        Returns:
        - List[genes]: List of gene subsets.
        """

        for i in range(self.n_repeats):
            seed = seed + i  # change seed per iteration
            train_genes, hidden_genes = train_test_split(self.genes_list,
                                                         test_size=0.2,
                                                         random_state=seed)
            self.folds_train.append(train_genes)
            self.folds_hidden.append(hidden_genes)

    def run_optuna(self):
        """
        Run Optuna with pruning and a single-objective: F1-score
        The pruning is defined.

        Returns:
        - pd.DataFrame or None: The final filtered graph or result.
        """
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = MedianPruner(
            n_startup_trials=1,
            n_warmup_steps=2,
            interval_steps=1
            )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

        self.generate_fixed_train()
        print(f"Trial number: {self.n_trials}")
        print(f"Repeats number: {self.n_repeats}")
        study.optimize(self.objective, self.n_trials, n_jobs=-1,
                       show_progress_bar=True)
        vis.plot_optimization_history(study).write_html("history.html")
        vis.plot_param_importances(study).write_html("importance.html")
        vis.plot_parallel_coordinate(study).write_html("parallel.html")
        vis.plot_intermediate_values(study).write_html("intermediate.html")
        vis.plot_slice(study).write_html("slice.html")
        vis.plot_contour(study).write_html("contour.html")
        vis.plot_edf(study).write_html("edf.html")

        print("Best value:", study.best_value)
        print("Best parameters:", study.best_params)

        self.analyse_best_trials(study)
        network_best = self.calculate_best_result(study.best_params)

        return network_best

    def calculate_best_result(self, params):
        """
        Calculate the result with the best parameters found by Optuna.
        Parameters:
        - parameters : parameters of the best trial.
        Returns:
        - DataFrame : the interactions present in the network.
        """
        results = NetworkAnalysisResults(self.genes_list)
        seed = 42
        for j in range(10):
            seed = seed + j  # change seed per iteration
            fold_genes, hidden_fold = train_test_split(self.genes_list,
                                                       test_size=0.2,
                                                       random_state=seed)
            fold_genes = list(fold_genes)
            self.build_graph(j, results, set(fold_genes),
                             params['confidence'],
                             params['n_genes_to_add'])
            results_total = results.compute_edge_stats_with_metadata(j+1)
        mask = results_total['pourcentage'] >= params["conservation_threshold"]
        filtered = results_total[mask].copy()
        genes_set = self.genes_set(filtered)
        print(genes_set)
        return filtered
