# master-thesis

## Project Description
Project focuses on the use of graph neural networks in tabular data.
It tests various approaches to feature graph construction and evaluates the performance of different GNN architectures.
Graph neural networks specyfic explanation methods are also applied to understand the model's decision-making process.

## Installation

### Requirements

- Python 3.12
- Poetry 2.3.4

Install the required Poetry version with `python -m pip install poetry==2.3.4`, then run `poetry install` to install the project.


```bash
python -m pip install poetry==2.3.4
```

Install the project and its dependencies from the repository root:

```bash
poetry install
```

### Environment configuration

Create a local `.env` file from the provided template using `cp .env_example .env`:

```bash
cp .env_example .env
```

Fill in every value in `.env` for your local environment:

| Variable | Description |
| --- | --- |
| `CONFIGS_DIR` | Absolute path to the project's `configs` directory. |
| `DATA_DIR` | Absolute path to the directory containing input and processed data. |
| `RESULTS_DIR` | Absolute path to the directory used for experiment outputs. |
| `RESULTS_LATEX_DIR` | Absolute path used for exported LaTeX tables and figures. |
| `OPTUNA_STORAGE_URL` | Optuna storage URL, such as a SQLite database URL. |
| `MLFLOW_TRACKING_URI` | URI of the MLflow tracking server. |

The `.env` file is ignored by Git and must not be committed.

### Pre-commit hooks

Install the repository hooks after installing the dependencies:

```bash
poetry run pre-commit install
```

## Notebooks

The tracked notebooks are organized by stage of the experimental workflow.

### `notebooks/data_cleaning`

- [`01_eda.ipynb`](notebooks/data_cleaning/01_eda.ipynb) - explores the raw medical dataset, including column statistics, missing values, distributions, and candidate target variables.
- [`02_cleaning.ipynb`](notebooks/data_cleaning/02_cleaning.ipynb) - cleans the medical data, derives variables, reduces categories, and saves the processed dataset.
- [`03_train_valid_test_split.ipynb`](notebooks/data_cleaning/03_train_valid_test_split.ipynb) - creates and saves stratified train, validation, test, and cross-validation splits.

### `notebooks/exploration`

- [`exploration_higgs.ipynb`](notebooks/exploration/exploration_higgs.ipynb) - explores the HIGGS dataset and calculates feature interaction-information matrices.
- [`exploration_medical.ipynb`](notebooks/exploration/exploration_medical.ipynb) - analyzes the medical cohort, feature distributions, missingness, and information-based interactions.
- [`interaction_recovery_analysis.ipynb`](notebooks/exploration/interaction_recovery_analysis.ipynb) - evaluates how accurately interaction information recovers known interactions in generated data.

### `notebooks/training`

- [`training_for_synthetic_data.ipynb`](notebooks/training/training_for_synthetic_data.ipynb) - tunes and compares GNN, MLP, and XGBoost models on synthetic and benchmark datasets.
- [`training_for_medical_data.ipynb`](notebooks/training/training_for_medical_data.ipynb) - tunes GNN, MLP, and XGBoost models using predefined medical-data cross-validation folds.
- [`final_refit.ipynb`](notebooks/training/final_refit.ipynb) - selects the best medical-data configurations, refits them, and evaluates them once on the held-out test set.

### `notebooks/evaluation`

- [`mlflow_results_evaluate_select_data.ipynb`](notebooks/evaluation/mlflow_results_evaluate_select_data.ipynb) - selects relevant MLflow runs and exports local run and graph-artifact snapshots.
- [`mlflow_results_evaluate_synthetic.ipynb`](notebooks/evaluation/mlflow_results_evaluate_synthetic.ipynb) - compares synthetic-experiment results across repeated data-generation seeds and exports plots and tables.
- [`mlflow_results_evaluate_medical.ipynb`](notebooks/evaluation/mlflow_results_evaluate_medical.ipynb) - combines medical cross-validation and final-refit results and analyzes interaction-graph stability.
