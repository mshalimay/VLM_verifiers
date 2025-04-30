import concurrent.futures
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import timing_utils as timer
from utils.file_utils import find_files
from utils.timing_utils import dump_timings, timeit

DOMAINS = ["shopping", "classifieds", "reddit"]
OVEWRITE = False

experiments_path = "./offline_experiments/"
original_scores_path = "./experiments/gpt-4o-2024-08-06/base_stateaware/shopping/consolidated_summary_data.csv"
eval_criteria = ["SUCCESS", "PARTIAL SUCCESS", "FAILURE"]
exclude_strs = ["zzOld", "logs", "gemini", "llama", "claude", "k-cond", "k-uncond"]
results_csv_name = "evaluations.csv"

EVAL_SECTION_REGEX = re.compile(
    r"(?:EVALUATION:|Status:|# EVALUATION)\s*(.*?)\s*(?:FEEDBACK:|</pre>)",
    re.S | re.IGNORECASE,
)

EVAL_CRITERIA_REGEX = re.compile(
    rf"^\s*:?\s*({'|'.join(map(re.escape, eval_criteria))})(?::)?",
    re.IGNORECASE,
)


def parse_eval(content: str) -> str:
    """
    Extracts the evaluation score (e.g., SUCCESS, FAILURE) from the EVALUATION section.
    """
    match = EVAL_CRITERIA_REGEX.search(content)
    if match:
        return match.group(1).upper()  # Normalize to uppercase
    else:
        return ""


def get_eval_from_trace_file(file: Path | str) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    task_id = get_task_id_from_trace_file(file)
    variation_name = get_variation_name_from_trace_file(file)
    try:
        timer.start("open_file")
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            timer.end("open_file")

            # Extract the evaluation section using a regex.
            timer.start("find_eval_section")
            matches = EVAL_SECTION_REGEX.findall(content)
            timer.end("find_eval_section")
            if not matches:
                print(f"No evaluation found in {file}", flush=True)
                # os.remove(file)
                return task_id, variation_name, ""

            # Get the last match
            eval_section_str = matches[-1].strip()
            timer.start("parse_eval")
            eval_result = parse_eval(eval_section_str)  # SUCESS, FAILURE, etc
            timer.end("parse_eval")

            if eval_result == "":
                print(f"Unable to parse textual score from eval section: {eval_section_str} for {file}", flush=True)
                # os.remove(file)
                return task_id, variation_name, ""

            return task_id, variation_name, eval_result
    except Exception as e:
        print(f"Error processing {file}: {e}", flush=True)
        return task_id, variation_name, ""


def get_task_id_from_trace_file(file: Path | str) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    return file.stem


def get_variation_name_from_trace_file(file: Path | str) -> str:
    if not isinstance(file, Path):
        file = Path(file)

    return file.parent.parent.name


def save_csv(
    df: pd.DataFrame, path: Path | str, gold_scores: pd.DataFrame | None = None, key_gold_score: str = "gold_score"
):
    df_to_save = df.copy()
    if gold_scores is not None:
        # Ensure task_id is of type string for both DataFrames
        gold_df = gold_scores.copy()
        gold_df["task_id"] = gold_df["task_id"].astype(str)
        df_to_save["task_id"] = df_to_save["task_id"].astype(str)

        # Subset gold_scores to only include task_id and score, and rename score to gold_score
        gold_scores_subset = gold_df[["task_id", key_gold_score]]

        # Drop the existing gold_score column if it already exists
        if key_gold_score in df_to_save.columns:
            df_to_save = df_to_save.drop(columns=[key_gold_score])

        df_to_save = pd.merge(df_to_save, gold_scores_subset, on="task_id", how="left")

    df_to_save.to_csv(path, index=False)


def remove_files_to_parse_if_exists(files_to_parse: list[Path], existing_results_df: pd.DataFrame) -> list[Path]:
    variation_names = [get_variation_name_from_trace_file(file) for file in files_to_parse]
    task_ids = [get_task_id_from_trace_file(file) for file in files_to_parse]

    final_files_to_parse = []

    # Create lookup dictionaries for faster access
    existing_values = {
        (str(row["task_id"]), col): row[col]
        for _, row in existing_results_df.iterrows()
        for col in existing_results_df.columns
        if col != "task_id"
    }

    # Create a list of files that need to be parsed
    for file, variation_name, task_id in zip(files_to_parse, variation_names, task_ids):
        if (task_id, variation_name) in existing_values:
            val = existing_values[(task_id, variation_name)]
            if val and pd.notna(val):
                continue
        final_files_to_parse.append(file)

    return final_files_to_parse


def parse_experiment_results(
    base_path,
    gold_scores: pd.DataFrame | None = None,
    key_gold_score: str = "gold_score",
    existing_results_df: pd.DataFrame | None = None,
):
    print(f"Parsing evaluations in {base_path}", flush=True)

    files_to_parse = list(Path(base_path).glob("**/*.html"))
    valid_filename_pattern = re.compile(r"^\d+\.html$")
    files_to_parse = [file for file in files_to_parse if valid_filename_pattern.match(file.name)]

    files_to_parse = [file for file in files_to_parse if not any(exclude in str(file) for exclude in exclude_strs)]

    if existing_results_df is not None:
        files_to_parse = remove_files_to_parse_if_exists(files_to_parse, existing_results_df)

    if len(files_to_parse) == 0:
        print(f"No files to parse in {base_path}", flush=True)
        return None

    results = {}

    # [get_eval_from_trace_file(file) for file in files_to_parse]  # @debugging

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Submit all file processing tasks concurrently.
        futures = {executor.submit(get_eval_from_trace_file, file): file for file in files_to_parse}
        for future in concurrent.futures.as_completed(futures):
            task_id, variation_name, eval_result = future.result()
            if not eval_result:
                print(f"Not able to parse eval result for {base_path}/{variation_name}/conversation/{task_id}.html")
                continue

            if task_id not in results:
                results[task_id] = {}
            results[task_id][variation_name] = eval_result
            # dump_timings(os.path.dirname(__file__))

    if not results:
        return existing_results_df

    # Convert dictionary to DataFrame after processing all subdirs
    result_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    result_df.rename(columns={"index": "task_id"}, inplace=True)
    result_df["task_id"] = result_df["task_id"].astype(str)

    result_csv_path = Path(base_path) / results_csv_name

    if existing_results_df is not None:
        result_df = result_df.set_index("task_id").combine_first(existing_results_df.set_index("task_id")).reset_index()

    save_csv(result_df, result_csv_path, gold_scores=gold_scores, key_gold_score=key_gold_score)
    return result_df


def map_eval_to_score(eval: str) -> int | float:
    # If eval is None, empty, or not provided, return np.nan so that it doesn't count.
    if not eval:  # This catches None or an empty string.
        return np.nan
    if eval == "SUCCESS":
        return 1
    elif eval == "PARTIAL SUCCESS":
        return 0
    elif eval == "FAILURE":
        return 0
    else:
        return np.nan


def compute_confusion_stats(gold_scores: pd.DataFrame, evals: pd.DataFrame):
    # Convert task_id to string
    gold_scores["task_id"] = gold_scores["task_id"].astype(str)
    evals["task_id"] = evals["task_id"].astype(str)

    evals = evals.copy()
    evals["predicted_score"] = evals["eval"].apply(map_eval_to_score)

    # Merge evals with the ground-truth scores on task_id
    merged = pd.merge(evals, gold_scores, on="task_id", how="right", suffixes=("_eval", "_true"))

    # Compute confusion matrix values based on the comparison of predicted vs. true
    false_positive = ((merged["predicted_score"] == 1) & (merged["gold_score"] == 0)).sum()
    false_negative = ((merged["predicted_score"] == 0) & (merged["gold_score"] == 1)).sum()
    true_positive = ((merged["predicted_score"] == 1) & (merged["gold_score"] == 1)).sum()
    true_negative = ((merged["predicted_score"] == 0) & (merged["gold_score"] == 0)).sum()

    tp_ratio = true_positive / (true_positive + false_negative)
    tn_ratio = true_negative / (false_positive + true_negative)
    fp_ratio = false_positive / (false_positive + true_negative)
    fn_ratio = false_negative / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Added counts for SUCCESS, PARTIAL SUCCESS, and FAILURE based on the "eval" column.
    criteria_counts = {}
    for criteria in eval_criteria:
        criteria_count = evals["eval"].str.upper().eq(criteria).sum()
        criteria_counts[criteria] = criteria_count

    # Filter the false negatives (i.e. cases where gold_score==1 but predicted 0)
    merged["eval_upper"] = merged["eval"].str.upper().str.strip()
    false_negatives = merged[(merged["predicted_score"] == 0) & (merged["gold_score"] == 1)]
    false_neg_partial_success = false_negatives[false_negatives["eval_upper"] == "PARTIAL SUCCESS"].shape[0]
    false_neg_failure = false_negatives[false_negatives["eval_upper"] == "FAILURE"].shape[0]

    total = len(gold_scores)

    all_data = {
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "NA": total - false_positive - false_negative - true_positive - true_negative,
        "tp_ratio": tp_ratio,
        "tn_ratio": tn_ratio,
        "fp_ratio": fp_ratio,
        "fn_ratio": fn_ratio,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "false_neg_partial_success": false_neg_partial_success,
        "false_neg_failure": false_neg_failure,
    }

    all_data.update(criteria_counts)

    return all_data


# ===============================================================
# Main
# ===============================================================
# Assumes folder structure:
# experiments/
#   gpt-4o-2024-08-06/
#     experiments_domain_1_date_1/
#       variation_1/
#       variation_2/
#       ...
#     experiments_domain_2_date_2/
#       variation_1/
#       variation_2/
#       ...


all_evals = pd.DataFrame()
gold_scores = pd.read_csv(original_scores_path)
gold_scores.rename(columns={"score": "gold_score"}, inplace=True)

# 1) Consolidate evals across [model, experiment, variations]  and across models into single csv files
for model_subdir in [subdir.name for subdir in Path(experiments_path).iterdir() if subdir.is_dir()]:
    if any(exclude in model_subdir for exclude in exclude_strs):
        continue

    # Get all experiment subdirs for this model.
    model_path = Path(experiments_path) / model_subdir
    experiments_subdirs = [subdir.name for subdir in Path(model_path).iterdir() if subdir.is_dir() and subdir.name]

    # Parse evaluations for each individual experiment subdir; saves `evaluations.csv` in each subdir with consolidated evals.
    for experiment_subdir in experiments_subdirs:  # shopping, reddit, etc
        experiment_path = model_path / experiment_subdir

        existing_results_df = None
        # Check if `evaluations.csv` already exists.
        if (experiment_path / results_csv_name).exists() and not OVEWRITE:
            existing_results_df = pd.read_csv(experiment_path / results_csv_name)
            existing_results_df["task_id"] = existing_results_df["task_id"].astype(str)

        results = parse_experiment_results(
            experiment_path, gold_scores=gold_scores, existing_results_df=existing_results_df
        )
        if results is None:
            continue

        # Append model name to col names
        rename_dict = {col: f"{col}--{model_subdir}" for col in results.columns if col != "task_id"}
        results.rename(columns=rename_dict, inplace=True)

        # Add source and domain columns
        results["source"] = experiment_subdir
        # Find which domain appears in experiment_subdir
        domain = next((d for d in DOMAINS if d in experiment_subdir), None)
        results["domain"] = domain

        # Merge results across experiments
        if all_evals.empty:
            all_evals = results
        else:
            all_evals["task_id"] = all_evals["task_id"].astype(str)
            results["task_id"] = results["task_id"].astype(str)
            all_evals = pd.merge(all_evals, results, on=["task_id", "domain", "source"], how="outer")

    # Pivot table to ensure a single row per task_id
    consolidated_evals = all_evals.pivot_table(index="task_id", columns="domain", aggfunc="first")

    # Flatten the column multi-index to a single level
    consolidated_evals.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in consolidated_evals.columns]
    consolidated_evals.reset_index(inplace=True)
    consolidated_evals["task_id"] = consolidated_evals["task_id"].astype(str)

# Save consolidated eval results
save_path = Path(experiments_path) / results_csv_name
save_csv(consolidated_evals, save_path, gold_scores=gold_scores)
print(f"Evaluation results consolidated for all models and experiments at {save_path}")


# 2) Compute confusion matrices.
confusion_matrices = pd.DataFrame()
consolidated_evals["task_id"] = consolidated_evals["task_id"].astype(str)
gold_scores["task_id"] = gold_scores["task_id"].astype(str)

for col in consolidated_evals.columns:
    if col == "task_id" or "source" in col or "gold_score" in col:
        continue

    # Prepare a DataFrame with task_id and the evaluation results.
    # Rename the column to "eval" so that our confusion_matrix function can apply map_eval_to_score.
    evals_subset = consolidated_evals[["task_id", col]].rename(columns={col: "eval"})

    # Compute the confusion matrix comparing model evaluations with the true scores.
    confusion_stats = compute_confusion_stats(gold_scores, evals_subset)

    # Remove any domain from the column name
    for domain in DOMAINS:
        if domain in col:
            col = re.sub(rf"{domain}_", "", col)

    # Column name is the col name, rows are the confusion matrix values
    confusion_matrices[col] = pd.Series(confusion_stats)


new_columns = []
for col in confusion_matrices.columns:
    # Expecting column names like:
    # "shopping_two_compare_actions_privileged--gpt-4o-2024-08-06__privileged_hint"
    if "--" in col:
        # Split the column name into left (case info) and right (model config)
        case_config, model_config = col.split("--", 1)

        new_columns.append((model_config, case_config))
    else:
        # In case any column doesn't have the delimiter, keep it as is:
        new_columns.append(("", col))

# Convert the list of tuples into a MultiIndex and assign to the DataFrame's columns.
confusion_matrices.columns = pd.MultiIndex.from_tuples(new_columns)

confusion_matrices = confusion_matrices.sort_index(axis=1, level=0)
confusion_matrices.to_csv(Path(experiments_path) / "confusion_matrices.csv", index=True)
print("Offline experiment stats saved to", Path(experiments_path) / "confusion_matrices.csv")
