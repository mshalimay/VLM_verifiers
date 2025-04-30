import os
import re

from bs4 import BeautifulSoup

from utils.debug_utils import set_env_variables
from utils.string_utils import clean_spaces

from .vwa_specific import get_interaction_history_message

set_env_variables()
cache_trajectory_msgs = {}


def get_trajectory_msgs(config, trace_data):
    prompt_config = config["prompt_args"]
    use_a = True if prompt_config["trace_info"] == "actions" else False
    use_u = True if prompt_config["trace_info"] == "utt" else False
    state_idxs = prompt_config.get("state_idxs", [])

    # If cached, return cached message
    cache_key = (use_a, use_u, tuple(state_idxs), trace_data["trajectory_path"])

    if cache_key in cache_trajectory_msgs:
        return cache_trajectory_msgs[cache_key]

    # Else, create messages, cache and return
    trajectory_msgs = get_interaction_history_message(
        trace_data=trace_data, actions=use_a, utterances=use_u, state_idxs=state_idxs
    )
    cache_trajectory_msgs[cache_key] = trajectory_msgs
    return trajectory_msgs


def parse_evaluation(response: str) -> list[str]:
    splitters = ["EVALUATION:", "FEEDBACK:"]
    utterances = []
    splitters_group = "|".join(map(re.escape, splitters))
    for splitter in splitters:
        pattern = rf"{splitter}(.*?)(?:\n|{splitters_group}|$)"
        # pattern = rf"(?:\*+|#+|\s*)({splitters_group}):\s*(.*?)(?=\n(?:\s*(?:\*+|#+|\s*)({splitters_group}):)|$)"

        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            utterances.append(clean_spaces(match.group(1)))
        else:
            utterances.append("error")
            # raise ValueError(f"Cannot find {splitter} in {response}")
    return utterances


def get_response_from_html_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""

    try:
        # Open and read the HTML file
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the last <pre> tag
        pre_tags = soup.find_all("pre")
        if pre_tags:
            last_pre_tag = pre_tags[-1]
            return last_pre_tag.text.strip()

        return "No <pre> tags found in the provided HTML file."

    except Exception as e:
        return ""
