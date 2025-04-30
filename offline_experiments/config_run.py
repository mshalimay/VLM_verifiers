import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from itertools import product

# Option 1: Ignore warnings whose message starts with "Pydantic serializer warnings:"
warnings.filterwarnings("ignore", message="Pydantic serializer warnings:")

from utils.signal_utils import signal_manager

CASES = {
    # "1p_no_k": {
    #     "eval_criterias": ["bin", "tri"],
    #     "cot_parts": ["no_cot", "basic_cot"],
    #     "trace_info": ["actions", "utt"],
    #     "add_summ_info": False,
    #     "add_expectation_info": False,
    # },
    # "1p_k": {
    #     "eval_criterias": ["bin", "tri"],
    #     "cot_parts": ["basic_cot", "compare", "desc_compare"],
    #     "trace_info": ["actions", "utt"],
    #     "add_summ_info": False,
    #     "add_expectation_info": True,
    # },
    "2p": {
        "eval_criterias": ["quad"],
        "cot_parts": ["basic_cot", "desc_compare"],
        "trace_info": ["utt"],
        "k_config": [
            # {"expert": False, "conditional": False, "cached_k_dir": None},
            {"expert": True, "conditional": False, "cached_k_dir": None},
            # {"conditional": True, "expert": False, "cached_k_dir": None},
        ],
        "add_summ_info": False,
        "add_expectation_info": True,
    },
}

gen_config = {
    "model": "gpt-4o-2024-08-06",
    # "model": "gemini-2.5-flash-preview-04-17",
    # "model": "gemini-2.0-flash-001",
    "temperature": 1,
    "top_p": 0.01,
    "top_k": 40,
    "max_tokens": 8192,
    "thinking_budget": 0,
    # "engine": "openai",
    # "metadata": {
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "provider": "openrouter",
    # },
}

run_config = {
    "overwrite": False,
    "batch_mode": True,
    "max_batch_size": 20 if "gemini" in gen_config["model"] else -1,
    "multiprocess_batch_mode": True if "gemini" in gen_config["model"] else False,
    "num_processes": 2,
}

# Task data to load
domain = "shopping"
base_trace_path = f"./experiments/gpt-4o-2024-08-06/base_stateaware/{domain}/htmls/render_{{task_id}}.html"
task_ids_file = "evaluation_harness/task_subsets/shopping.txt"

with open(task_ids_file, "r") as file:
    task_ids = [line.strip() for line in file.readlines() if line.strip().isnumeric()]

if sys.gettrace():  # @debug
    # task_ids = task_ids[-5:]
    task_ids = ["315"]


timestamp = datetime.now().strftime("%Y-%m-%d")
exper_dir = "./offline_experiments"
model_name = re.sub("/", "-", gen_config["model"].split("/")[-1])
output_dir = f"{exper_dir}/{model_name}/{domain}"


def dump_exper_args(config: dict = None):
    if os.path.exists(f"{output_dir}/exper_args.json"):
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/exper_args.json", "w") as f:
        args_to_dump = {
            "gen_config": gen_config,
            "trace_path": base_trace_path,
            "task_ids": task_ids,
            "date": timestamp,
        }

        if config is not None:
            args_to_dump.update(config)

        json.dump(args_to_dump, f, indent=4)


def get_first_pass_subdir_name(config):
    trace_info = config["prompt_args"]["trace_info"]
    k_config = config["prompt_args"]["k_config"]
    dir_str = "k-"
    if k_config["conditional"]:
        dir_str += "cond-"
        if trace_info:
            dir_str += f"{trace_info}-"
    else:
        dir_str += "uncond-"

    dir_str += "expert" if k_config["expert"] else ""
    dir_str = dir_str.strip("-")
    return dir_str


def build_config_name(eval_criteria="", cot_part="", trace_info="", k_config=None):
    config_str = ""
    if eval_criteria:
        config_str += f"{eval_criteria}"
    if cot_part:
        config_str += f"-{cot_part}"
    if trace_info:
        config_str += f"-{trace_info}"

    k_config_str = ""
    if k_config:
        k_config_str += "-cond" if k_config["conditional"] else "-uncond"
        k_config_str += "-expert" if k_config["expert"] else ""

    # Strip leading "-"
    config_str = config_str.lstrip("-")
    k_config_str = k_config_str.lstrip("-")
    final_str = f"{config_str}-{k_config_str}".lstrip("-")

    return final_str


def build_all_first_pass_configs(output_dir=output_dir):
    all_configs = {}
    for case_name, case_config in CASES.items():
        if "k_config" not in case_config:
            continue

        combinations = []
        for k_config in case_config["k_config"]:
            # Check if this is a conditional case
            is_conditional = k_config["conditional"]

            if is_conditional:
                # Generate combinations for conditional cases (original behavior)
                combinations.extend(
                    list(
                        product(
                            case_config["trace_info"],
                            [k_config],
                        )
                    )
                )
            else:
                # For unconditional cases, ignore trace_info combinations
                combinations.append((None, k_config))

        # Create new cases with all combinations
        for i, (trace_info, k_config) in enumerate(combinations):
            key = f"{case_name}-"
            key += build_config_name(trace_info=trace_info, k_config=k_config)

            config = {
                "prompt_args": {
                    "eval_criteria": "",
                    "cot_part": "",
                    "trace_info": trace_info,
                    "add_summ_info": case_config["add_summ_info"],
                    "add_expectation_info": case_config["add_expectation_info"],
                    "k_config": k_config,
                },
            }
            out_dir = f"{output_dir}/{get_first_pass_subdir_name(config)}"
            config["out_dir"] = out_dir
            all_configs[key] = config
    return all_configs


def build_all_eval_configs(output_dir=output_dir):
    all_configs = {}
    for case_name, case_config in CASES.items():
        # Generate all combinations for each case
        combinations = list(
            product(
                case_config["eval_criterias"],
                case_config["cot_parts"],
                case_config["trace_info"],
                case_config.get("k_config", [None]),
            )
        )

        # Create new cases with all combinations
        for i, (eval_criteria, cot_part, trace_info, k_config) in enumerate(combinations):
            key = f"{case_name}-"
            key += build_config_name(eval_criteria, cot_part, trace_info, k_config)

            all_configs[key] = {
                "prompt_args": {
                    "eval_criteria": eval_criteria,
                    "cot_part": cot_part,
                    "trace_info": trace_info,
                    "add_summ_info": case_config["add_summ_info"],
                    "add_expectation_info": case_config["add_expectation_info"],
                    "k_config": k_config.copy() if k_config else None,
                },
                "out_dir": f"{output_dir}/{case_name}-{build_config_name(eval_criteria, cot_part, trace_info, k_config)}",
            }
            if k_config:
                if k_config.get("cached_k_dir"):
                    topmost_dir = k_config["cached_k_dir"]
                    cached_k_dir = f"{topmost_dir}/{get_first_pass_subdir_name(all_configs[key])}"
                else:
                    cached_k_dir = f"{output_dir}/{get_first_pass_subdir_name(all_configs[key])}"
                all_configs[key]["prompt_args"]["k_config"]["cached_k_dir"] = cached_k_dir
    return all_configs


def run_and_wait(command, log_file_path):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    signal_manager.register_termination_signals(lambda: process.terminate())

    while True:
        output_line = process.stdout.readline()
        if output_line == "" and process.poll() is not None:
            break
        if output_line:
            print(output_line, end="")  # Print to command line
            logfile.write(output_line)
            logfile.flush()
    process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", action="store_true")
    parser.add_argument("--v", action="store_true")
    args = parser.parse_args()

    timestamp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"{exper_dir}/logs/{model_name}-{timestamp_time}.log"

    os.makedirs(f"{exper_dir}/logs", exist_ok=True)

    if args.fp:
        with open(log_file_path, "w") as logfile:
            run_and_wait(
                ["python", "-u", "-m", "offline_experiments.first_pass"],
                logfile,
            )

    if args.v:
        with open(log_file_path, "a") as logfile:
            run_and_wait(
                ["python", "-u", "-m", "offline_experiments.verify"],
                logfile,
            )
