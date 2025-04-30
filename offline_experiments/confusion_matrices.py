import asyncio
import concurrent.futures
import re
from pathlib import Path

import aiofiles
import numpy as np
import pandas as pd

DOMAINS = ["shopping", "classifieds", "reddit"]

experiments_path = "./offline_experiments/critique"
original_scores_path = "./experiments/gpt-4o-2024-08-06/base_stateaware/shopping/consolidated_summary_data.csv"
eval_criteria = ["SUCCESS", "PARTIAL SUCCESS", "FAILURE"]
exclude_subdirs = ["zzOld", "gemini", "llama", "claude"]
results_csv_name = "evaluations.csv"
OVEWRITE = True


def parse_eval(content: str) -> str:
    """
    Extracts the evaluation score (e.g., SUCCESS, FAILURE) from the EVALUATION section.
    """
    eval_scores = "|".join(map(re.escape, eval_criteria))  # Join criteria with OR operator
    status_pattern = rf"(?i)\s*({eval_scores})"

    match = re.search(status_pattern, content, re.IGNORECASE)
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
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error processing {file}: {e}", flush=True)
        return task_id, variation_name, ""

    # Extract the evaluation section using a regex.
    matches = re.findall(
        r"(?:EVALUATION:|Status:|# EVALUATION)\s*(.*?)\s*(?:FEEDBACK:|</pre>)", content, re.S | re.IGNORECASE
    )
    if not matches:
        print(f"No evaluation found in {file}", flush=True)
        return task_id, variation_name, ""

    # Choose the last match
    eval_section_str = matches[-1].strip()
    eval_result = parse_eval(eval_section_str)  # SUCESS, FAILURE, etc

    if eval_result == "":
        print(f"Unable to parse textual score from eval section: {eval_section_str} for {file}", flush=True)
        return task_id, variation_name, ""

    return task_id, variation_name, eval_result


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
        df_to_save = pd.merge(df_to_save, gold_scores_subset, on="task_id", how="left")

    df_to_save.to_csv(path, index=False)


def parse_experiment_results(base_path, gold_scores: pd.DataFrame | None = None, key_gold_score: str = "gold_score"):
    print(f"Parsing evaluations in {base_path}", flush=True)

    exclude_strs = ["knowledge_retrieval_uncond", "knowledge_retrieval_cond", "summarization"]

    files_to_parse = list(Path(base_path).glob("**/*.html"))
    valid_filename_pattern = re.compile(r"^\d+\.html$")
    files_to_parse = [file for file in files_to_parse if valid_filename_pattern.match(file.name)]

    files_to_parse = [file for file in files_to_parse if not any(exclude in str(file) for exclude in exclude_strs)]
    if len(files_to_parse) == 0:
        print(f"No files to parse in {base_path}", flush=True)
        return None

    results = {}

    # #  [get_eval_from_trace_file(file) for file in files_to_parse] sequential for debugging
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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

    # Convert dictionary to DataFrame after processing all subdirs
    result_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    result_df.rename(columns={"index": "task_id"}, inplace=True)
    result_df["task_id"] = result_df["task_id"].astype(str)

    result_csv_path = Path(base_path) / results_csv_name
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
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Added counts for SUCCESS, PARTIAL SUCCESS, and FAILURE based on the "eval" column.
    criteria_counts = {}
    for criteria in eval_criteria:
        criteria_count = evals["eval"].str.upper().eq(criteria).sum()
        criteria_counts[criteria] = criteria_count

    # Calculate overall accuracy if any records remain
    total = len(gold_scores)

    all_data = {
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "NA": total - false_positive - false_negative - true_positive - true_negative,
        "tp_ratio": tp_ratio,
        "tn_ratio": tn_ratio,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
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
    if any(exclude in model_subdir for exclude in exclude_subdirs):
        continue

    # Get all experiment subdirs for this model.
    model_path = Path(experiments_path) / model_subdir
    experiments_subdirs = [subdir.name for subdir in Path(model_path).iterdir() if subdir.is_dir() and subdir.name]

    # Parse evaluations for each individual experiment subdir; saves `evaluations.csv` in each subdir with consolidated evals.
    for experiment_subdir in experiments_subdirs:
        experiment_path = model_path / experiment_subdir

        # Check if `evaluations.csv` already exists.
        if (experiment_path / results_csv_name).exists() and not OVEWRITE:
            results = pd.read_csv(experiment_path / results_csv_name)
        else:
            results = parse_experiment_results(experiment_path, gold_scores=gold_scores)
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
print("Confusion matrices saved to", Path(experiments_path) / "confusion_matrices.csv")
