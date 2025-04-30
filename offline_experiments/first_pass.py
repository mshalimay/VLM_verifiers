import asyncio
import os
import sys

from utils.logger_utils import logger

if __name__ == "__main__" and not __package__:
    # Insert the parent directory into sys.path so that the package can be found
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)
    # Manually set the package name so that relative imports work
    __package__ = "offline_experiments"

from .build_prompt import get_sys_prompt_verifier, k_retrieval_2p, k_retrieval_expert, sys_prompt_k_expert
from .config_run import (
    base_trace_path,
    build_all_first_pass_configs,
    gen_config,
    get_first_pass_subdir_name,
    run_config,
    task_ids,
)
from .runners import run_batch_mode, run_sequential
from .utils_offline_exper import get_response_from_html_file, get_trajectory_msgs
from .vwa_specific import get_intent_message, get_trace_data

cache_k_responses = {}


def _get_cache_key(trace_data, config):
    return (trace_data["task_id"], config["prompt_args"]["trace_info"], str(config["prompt_args"]["k_config"]))


def _get_k_from_cache(trace_data, config, cache_key: str):
    try:
        if cache_key and cache_key in cache_k_responses:
            return cache_k_responses[cache_key]

        if trace_data and config:
            cache_key = _get_cache_key(trace_data, config)
            if cache_key in cache_k_responses:
                return cache_k_responses[cache_key]

        return None
    except Exception as _:
        return None


def get_k_resp_from_cache(
    trace_data=None,
    config=None,
    file_path: str = "",
):
    # Try to get K from cache
    k_resp = _get_k_from_cache(trace_data, config, file_path)
    if k_resp:
        return k_resp

    if file_path:
        k_resp = get_response_from_html_file(file_path)
        cache_k_responses[file_path] = k_resp
        return k_resp

    if not trace_data or not config:
        return None

    k_dir_name = get_first_pass_subdir_name(trace_data, config["prompt_args"]["k_config"])
    k_path = f"{config['out_dir']}/{k_dir_name}/{trace_data['task_id']}"
    k_resp = get_response_from_html_file(k_path)
    if k_resp:
        cache_key = _get_cache_key(trace_data, config)
        cache_k_responses[cache_key] = k_resp
    return k_resp


def build_prompt_first_pass(
    trace_data,
    config,
    trajectory_msgs=None,
):
    prompt_config = config["prompt_args"]
    k_prompt_config = prompt_config["k_config"]
    conditional: bool = k_prompt_config["conditional"]
    expert: bool = k_prompt_config["expert"]

    if conditional and not trajectory_msgs:
        raise ValueError("Trajectory msgs are required for conditional K retrieval")

    if expert:
        k_retrieval_query = k_retrieval_expert
        _sys_prompt = sys_prompt_k_expert
    else:
        k_retrieval_query = k_retrieval_2p
        _sys_prompt = get_sys_prompt_verifier(
            trace_info=config["prompt_args"]["trace_info"],
            add_summ_info=config["prompt_args"]["add_summ_info"],
            add_web_knowledge_info=config["prompt_args"]["add_expectation_info"],
            two_pass=True,
        )

    if conditional:
        add_state_idxs = []
        state_img_intros = []
    else:
        add_state_idxs = [0]
        state_img_intros = ["Initial webpage screenshot"]

    objective_msg = get_intent_message(
        trace_data=trace_data,
        add_state_idxs=add_state_idxs,
        state_img_intros=state_img_intros,
    )

    if conditional:
        prompt = [{"role": "system", "content": _sys_prompt}, objective_msg, trajectory_msgs, k_retrieval_query]
    else:
        prompt = [{"role": "system", "content": _sys_prompt}, objective_msg, k_retrieval_query]

    return prompt


def build_llm_call_args(task_id, config, run_config) -> tuple[list[dict], str, str, str]:
    try:
        conversation_dir = f"{config['out_dir']}/conversation"
        usage_dir = f"{config['out_dir']}/usage"
        if not run_config["overwrite"]:
            full_conversation_path = f"{conversation_dir}/{task_id}.html"
            if os.path.exists(full_conversation_path):
                logger.info(f"Skipping {task_id}: {full_conversation_path} exists.")
                return None, None, None, None

        k_config = config["prompt_args"].get("k_config", None)
        if not k_config:
            return None, None, None, None

        trace_data = get_trace_data(base_trace_path.format(task_id=task_id), task_id)
        trajectory_msgs = get_trajectory_msgs(config, trace_data)
        prompt = build_prompt_first_pass(trace_data, config, trajectory_msgs)

        logger.info(f"FIRST PASS: Building llm call args for task {task_id}, config {config}")
        return prompt, conversation_dir, usage_dir, task_id
    except Exception as e:
        logger.warning(f"FIRST PASS: Error creating llm call args for task {task_id}, config {config}: {e}")
        return None, None, None, None


if __name__ == "__main__":
    batch_mode = run_config["batch_mode"]
    all_configs = build_all_first_pass_configs()

    if len(all_configs) == 0:
        logger.info("No configs to run")
        exit()

    if batch_mode:
        logger.info("Running first pass in batch mode")
        run_batch_mode(
            all_configs,
            task_ids=task_ids,
            run_config=run_config,
            gen_config=gen_config,
            build_llm_call_args_fn=build_llm_call_args,
        )
    else:
        logger.info("Running first pass in sequential mode")
        asyncio.run(
            run_sequential(
                all_configs,
                task_ids=task_ids,
                run_config=run_config,
                gen_config=gen_config,
                build_llm_call_args_fn=build_llm_call_args,
            )
        )
