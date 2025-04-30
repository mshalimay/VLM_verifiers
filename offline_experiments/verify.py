import asyncio
import json
import os
import sys

from llms.llm_utils import visualize_prompt
from utils.logger_utils import logger

if __name__ == "__main__" and not __package__:  # @debug
    # Insert the parent directory into sys.path so that the package can be found
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)
    # Manually set the package name so that relative imports work
    __package__ = "offline_experiments"


from .build_prompt import get_verifier_prompts
from .config_run import base_trace_path, build_all_eval_configs, gen_config, output_dir, run_config, task_ids
from .first_pass import get_k_resp_from_cache
from .runners import run_batch_mode, run_sequential
from .utils_offline_exper import get_trajectory_msgs
from .vwa_specific import get_intent_message, get_trace_data

cache_trajectory_msgs = {}

# ===============================================
# Task data to load
# ===============================================

# Dump experiment args
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/exper_args.json", "w") as f:
    json.dump(gen_config, f, indent=4)


def build_llm_call_args(task_id, config, run_config) -> tuple[list[dict], str, str, str]:
    try:
        conversation_dir = f"{config['out_dir'].strip('-')}/conversation"
        usage_dir = f"{config['out_dir'].strip('-')}/usage"

        if not run_config["overwrite"]:
            full_conversation_path = f"{conversation_dir}/{task_id}.html"
            if os.path.exists(full_conversation_path):
                logger.info(f"Skipping {task_id} because {full_conversation_path} exists.")
                return None, None, None, None

        k_resp = None
        if config["prompt_args"].get("k_config"):
            k_dir_name = config["prompt_args"]["k_config"]["cached_k_dir"]
            k_resp = get_k_resp_from_cache(file_path=f"{k_dir_name}/conversation/{task_id}.html")
            if not k_resp:
                logger.error(f"Failed to build prompt for {config} {task_id}: unable to get first pass response.")
                return None, None, None, None

            # TODO: generate when not cached

        trace_path = base_trace_path.format(task_id=task_id)
        trace_data = get_trace_data(trace_path, task_id)
        if not trace_data:
            logger.error(f"Failed to build prompt for {config} {task_id}: unable to get trace data.")
            return None, None, None, None

        msg_intent = get_intent_message(trace_data=trace_data)
        trajectory_msgs = get_trajectory_msgs(config, trace_data)

        verifier_prompts = get_verifier_prompts(**config["prompt_args"])
        sys_prompt, eval_prompt = verifier_prompts["sys_prompt"], verifier_prompts["eval_prompt"]

        full_prompt = [{"role": "system", "content": sys_prompt}, msg_intent, trajectory_msgs, k_resp, eval_prompt]

        logger.info(f"VERIFY: Finished building llm call args for task {task_id}, config {config}")
        return full_prompt, conversation_dir, usage_dir, task_id
    except Exception as e:
        logger.error(f"Failed to build prompt for {config} {task_id}: {e}")
        return None, None, None, None


if __name__ == "__main__":
    batch_mode = run_config["batch_mode"]
    all_configs = build_all_eval_configs()

    if len(all_configs) == 0:
        logger.info("No configs to run")
        exit()

    if batch_mode:
        logger.info("Running verify in batch mode")
        run_batch_mode(
            all_configs,
            task_ids=task_ids,
            run_config=run_config,
            gen_config=gen_config,
            build_llm_call_args_fn=build_llm_call_args,
        )
    else:
        logger.info("Running verify in sequential mode")
        asyncio.run(
            run_sequential(
                all_configs,
                task_ids=task_ids,
                run_config=run_config,
                gen_config=gen_config,
                build_llm_call_args_fn=build_llm_call_args,
            )
        )
