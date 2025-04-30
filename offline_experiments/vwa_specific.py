# ===============================================
# VWA-specific
# ===============================================

import json
from typing import Any

from utils.debug_utils import set_env_variables

set_env_variables()

from agent.request_refiner import RequestRefiner

request_refiner = None
caption_image_fn = None
prompt_constructor = None

updated_task_intents = {}


def get_captioning_fn():
    import argparse

    from vwa_utils.captioner_utils import define_captioning_fn

    args = argparse.Namespace(
        agent_captioning_model="Salesforce/blip2-flan-t5-xl",
        agent_captioning_model_device="server-cuda",
        eval_captioning_model="Salesforce/blip2-flan-t5-xl",
        eval_captioning_model_device="server-cuda",
        observation_type="image_som",
    )
    caption_image_fn, eval_caption_image_fn = define_captioning_fn(args)
    return caption_image_fn, eval_caption_image_fn


def build_prompt_constructor(
    name_user="user",
    name_assistant="assistant",
    img_detail="auto",
    prompt_file="p_testing",
    prompt_constructor_class="PromptConstructor",
    text_first=True,
):
    import importlib

    import agent
    import agent.agent
    import agent.prompt_constructor

    importlib.reload(agent)
    importlib.reload(agent.agent)
    importlib.reload(agent.prompt_constructor)

    # Dynamically import the specified prompt constructor class
    prompt_constructor_module = importlib.import_module("agent.prompt_constructor")
    PromptConstructorClass = getattr(prompt_constructor_module, prompt_constructor_class)

    lm_config_dict = {
        "name_user": name_user,
        "name_assistant": name_assistant,
        "img_detail": img_detail,
        "text_first": text_first,
    }

    agent_config = {
        "prompt": prompt_file,
        "lm_config": lm_config_dict,
    }

    global prompt_constructor
    prompt_constructor = PromptConstructorClass(lm_config=lm_config_dict, agent_config=agent_config)
    return prompt_constructor


def get_config_base_dir_from_txt(txt_path) -> str:
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip().isdigit():
                return line.strip()
    return ""


def get_updated_task_intents(domain: str) -> dict[str, str]:
    global updated_task_intents
    if updated_task_intents.get(domain):
        return updated_task_intents[domain]

    else:
        updated_task_intents[domain] = {}

    dir_adjusted_tasks = "./evaluation_harness/tasks_adjusted"
    file_path = f"{dir_adjusted_tasks}/{domain}_vague.txt"
    test_config_dir = f"./config_files/vwa_not_vague/test_{domain}"
    with open(file_path, "r") as f:
        data = f.readlines()

    task_ids = []
    for line in data:
        if line.strip().isdigit():
            task_ids.append(line.strip())

    for task_id in task_ids:
        with open(f"{test_config_dir}/{task_id}.json", "r") as f:
            data = json.load(f)
            updated_task_intents[domain][task_id] = data["intent"]
    return updated_task_intents[domain]


def _update_task_intent(task_id: str, domain: str):
    _updated_task_intents = get_updated_task_intents(domain)
    if task_id in _updated_task_intents:
        return _updated_task_intents[task_id]
    return None


def get_domain_from_path(
    test_config_dir: str, domains=["shopping", "reddit", "classifieds", "gitlab", "maps", "shopping_admin"]
):
    for domain in domains:
        if domain in test_config_dir.lower():
            return domain
    return None


def get_trace_data(
    trajectory_path: str,
    task_id: str | int,
    update_task_intent: bool = True,
):
    from vwa_utils.extract_trajectory_html import extract_trajectory_data

    try:
        objective, trajectory, meta_data = extract_trajectory_data(trajectory_path)

        if update_task_intent:
            new_task_intent = _update_task_intent(task_id, get_domain_from_path(trajectory_path))
            if new_task_intent:
                objective["text"] = new_task_intent

        return {
            "objective": objective,
            "trajectory": trajectory,
            "meta_data": meta_data,
            "task_id": str(task_id),
            "trajectory_path": trajectory_path,
        }

    except Exception as e:
        print(f"Error extracting trajectory data from {trajectory_path}: {e}")
        return None


def get_prompt_constructor(
    name_user="user",
    name_assistant="assistant",
    img_detail="auto",
    prompt_file="p_testing",
    prompt_constructor_class="PromptConstructor",
    text_first=True,
):
    global prompt_constructor
    if not prompt_constructor:
        build_prompt_constructor(
            name_user=name_user,
            name_assistant=name_assistant,
            img_detail=img_detail,
            prompt_file=prompt_file,
            prompt_constructor_class=prompt_constructor_class,
            text_first=text_first,
        )
    return prompt_constructor


def get_intent_message(
    trace_data: dict[str, Any],
    add_state_idxs: list[int] = [],
    state_img_intros: list[str] = [],
):
    global caption_image_fn, request_refiner, prompt_constructor
    if not caption_image_fn:
        caption_image_fn, _ = get_captioning_fn()

    if not request_refiner:
        request_refiner = RequestRefiner(
            agent_config={},
            captioning_fn=caption_image_fn,
        )

    trajectory = trace_data["trajectory"]
    intent = trace_data["objective"]

    task_id = str(trace_data["task_id"])
    prompt_constructor = get_prompt_constructor()

    meta_data = {"task_id": task_id}
    request_refiner.next_action(trajectory, intent["text"], intent["images"], meta_data)

    objective_image_captions = prompt_constructor.get_image_captions(intent["images"], meta_data)

    return prompt_constructor.build_intent_message(
        trajectory,
        f"## OBJECTIVE:\n{intent['text']}",
        intent["images"],
        {},
        add_states_idxs=add_state_idxs,
        state_img_intros=state_img_intros,
        objective_image_captions=objective_image_captions,
        add_state_img=True,
    )


def get_interaction_history_message(
    trace_data: dict[str, Any],
    actions: bool = False,
    utterances: bool = True,
    state_idxs: list[int] = [],
):
    global prompt_constructor
    prompt_constructor = get_prompt_constructor()

    trajectory = trace_data["trajectory"]
    meta_data = trace_data["meta_data"]

    if actions:
        prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = True
    else:
        prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = False

    if utterances:
        prompt_constructor.instruction["meta_data"]["use_assistant_utterance"] = True
    else:
        prompt_constructor.instruction["meta_data"]["use_assistant_utterance"] = False

    if not state_idxs:
        state_idxs = list(range(len(trajectory.states)))

    interaction_history_msgs = prompt_constructor.build_interaction_history(
        trajectory, meta_data, idxs_history=state_idxs
    )
    return interaction_history_msgs
