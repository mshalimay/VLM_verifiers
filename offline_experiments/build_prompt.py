# fmt: off

# ===============================================
# LINK: System Prompt parts
# ===============================================

sys_prompt_critic_base = f"""You are an intelligent agent tasked with supervising an assistant navigating a web browser to accomplish a web-based task. Your job is to evaluate the assistant's work, and provide feedback so it can progress towards the objective.
\n
## Here's the information you'll have:
### The objective: This is the task the assistant is trying to complete.
### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
{{trace_info}}{{summary_info}}{{web_knowledge_info}}
\n
## Assistant's capabilities: To effectively analyze the assistant's work, consider the actions it can perform. These actions fall into the following categories:
### Page Operation Actions:
```click [id]```: Click on an element with a specific id on the webpage.
```type [id] [content] [enter_after]```: Type the content into the field with id. If `enter_after` is 0, the "Enter" key is not pressed after typing; otherwise, the "Enter" key is automatically pressed.
```hover [id]```: Hover over an element with id.
```press [key_comb]```: Press a keyboard key or key combination (e.g., delete, ctrl+a).
```scroll [down]``` or ```scroll [up]```: Scroll the webpage up or down.

### Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

### URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

### Completion Action:
```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible, issue this action with optionally a reason why.
{{privileged_hint}}\n
## To be successful, it is very important to follow the following rules:
1. You must not not assume the assistant's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.{{web_knowledge_rule}}
"""

trace_info_no_assistant = "\n### The execution trace: This is a sequence of webpage screenshots, detailing the web navigation so far."
trace_info_utt = "\n### The execution trace: This is a sequence of webpage screenshots paired with the assistant's responses, detailing the web navigation so far."
trace_info_actions = "\n### The execution trace: This is a sequence of webpage screenshots paired with the assistant's actions, detailing the web navigation so far."

summ_info = "\n### Summary of the execution trace: This is a textual summary of the navigation so far."

web_knowledge_info = "\n### General web knowledge: This is a general description of how tasks like this are typically accomplished on the web."
web_knowledge_rule = "\n3. Use the General web knowledge as a guide, but also consider the context of the specific task given to you."
web_knowledge_rule_1p = "\n3. Come up with a general description of how tasks like this are typically accomplished on the web.\n4. Use this general web knowledge as a guide, but also consider the context of the specific task given to you."



sys_prompt_k_expert = f"""You are an expert on web navigation. Your job is to provide a general description of how tasks like the ones provided to you are typically accomplished on the web.
\n
## Here's the information you'll have:
### Objective: This is an english description of the task, possibly accompanied by images that provide more context on what must be accomplished.
### Screenshots: These are screenshots of the webpage, giving you context of the current state of the navigation process."""

# ===============================================
# LINK: Evaluation Criterias
# ===============================================
eval_criteria_tri = """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
"""


eval_criteria_bin = """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
"""


eval_criteria_quad = """
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
PARTIAL FAILURE: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
"""

# ===============================================
# LINK: CoT formats
# ===============================================
no_cot = ""

basic_cot = """
REASONING: [Step by step reasoning to come up with your evaluation and feedback]
"""

desc_cot = """
EXECUTION TRACE DESCRIPTION: [Understand and describe the assistant's work]
REASONING: [Step by step reasoning to come up with your evaluation and feedback]
"""

compare = """
COMPARISON: [Compare the assistant's work with the general web knowledge. Make a step by step comparison of what the assistant did and what is expected.]
RELEVANCE: [Which of the steps missing are relevant in this context. If the steps are not relevant, explain why.]
CONCLUSION: [Based on the assistant's work, comparison to general web knowledge, and the objective, what are your conclusions?]
"""

desc_compare = """
EXECUTION TRACE DESCRIPTION: [Understand and describe the assistant's work]
COMPARISON: [Compare the assistant's work with the general web knowledge. Make a step by step comparison of what the assistant did and what is expected.]
RELEVANCE: [Which of the steps missing are relevant in this context. If the steps are not relevant, explain why.]
CONCLUSION: [Based on the assistant's work, comparison to general web knowledge, and the objective, what are your conclusions?]
"""

# ===============================================
# LINK: Evaluation request
# ===============================================

response_format = f"""
{{k_retrieval_step}}
{{cot_parts}}
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the assistant can progress towards the objective]"""

eval_prompt_template = f"""Now please provide your response.
\n
## Here is the evaluation criteria:
{{eval_criteria}}
\n
## Provide your response as follows:
{{response_format}}
"""

# ===============================================
# LINK: K retrieval request
# ===============================================
# K injection / retrieval
k_retrieval_2p = """Please first provide the following: 
[Description of how tasks such as this are typically accomplished on the web.]"""

k_retrieval_1p = """GENERAL WEB KNOWLEDGE: [Description of how tasks such as this are typically accomplished on the web.]"""

k_retrieval_expert = """Now please provide your response:
[Description of how tasks such as this are typically accomplished on the web.]"""


k_injection = """## General web knowledge:
{k}"""



# ===============================================
# Prompt mappings
# ===============================================

eval_criterias = {
    "bin": eval_criteria_bin,
    "tri": eval_criteria_tri,
    "quad": eval_criteria_quad,
}

cot_parts = {
    "no_cot": no_cot,
    "basic_cot": basic_cot,
    "compare": compare,
    "desc_compare": desc_compare,
}

trace_infos = {
    "no_assistant": trace_info_no_assistant,
    "utt": trace_info_utt,
    "actions": trace_info_actions,
}

def safe_format(string_template: str, fill_with: str = "", **kwargs) -> str:
    """
    Formats a given template using the provided keyword arguments.
    Missing keys in the template are replaced with an empty string.

    Args:
        template (str): The string template with placeholders.
        **kwargs: Key-value pairs for formatting.

    Returns:
        str: The formatted string with missing keys as empty strings.
    """

    class DefaultDict(dict):
        def __missing__(self, key):
            return fill_with

    return string_template.format_map(DefaultDict(**kwargs))


def get_query_prompt_eval(
    eval_criteria: str = "", 
    cot_part: str = "", 
    add_retrieval_step: bool = False,
) -> str:

    # Get CoT parts
    if cot_part:
        _cot_parts = cot_parts[cot_part].strip()
    else:
        _cot_parts = ""

    # Get eval criteria
    if eval_criteria:
        _eval_criteria = eval_criterias[eval_criteria].strip()
    else:
        _eval_criteria = ""

    # If one-pass, but retrieving K, add K retrieval step before CoT
    k_step = k_retrieval_1p.strip() if add_retrieval_step else ""

    _response_format = response_format.format(k_retrieval_step=k_step, cot_parts=_cot_parts).strip()

    # Get eval prompt
    eval_prompt = safe_format(
        eval_prompt_template,
        eval_criteria=_eval_criteria,
        response_format=_response_format,
    )
    return eval_prompt.strip()


def get_sys_prompt_verifier(
    trace_info: str = "", 
    add_summ_info: bool = False, 
    add_web_knowledge_info: bool = False,
    two_pass: bool = False,
) -> str:
    if trace_info:
        _trace_info = trace_infos[trace_info].strip()
    else:
        _trace_info = ""

    _summ_info = summ_info if add_summ_info else ""

    if add_web_knowledge_info:
        if two_pass:
            _web_knowledge_info = web_knowledge_info
            _web_knowledge_rule = web_knowledge_rule
        else:
            _web_knowledge_info = ""
            _web_knowledge_rule = web_knowledge_rule_1p
    else:
        _web_knowledge_info = ""
        _web_knowledge_rule = ""
    
    sys_prompt = safe_format(
        sys_prompt_critic_base,
        trace_info=_trace_info,
        summ_info=_summ_info,
        web_knowledge_info=_web_knowledge_info,
        web_knowledge_rule=_web_knowledge_rule,
    )
    return sys_prompt.strip()


def get_verifier_prompts(
    eval_criteria: str = "", 
    cot_part: str = "", 
    trace_info: str = "", 
    add_summ_info: bool = False, 
    add_expectation_info: bool = False, 
    **kwargs,
) -> dict[str, str]:
    if kwargs.get("k_config"):
        two_pass = True
    else:
        two_pass = False

    add_retrieval_step: bool = True if (add_expectation_info and not two_pass) else False

    eval_prompt = get_query_prompt_eval(eval_criteria, cot_part, add_retrieval_step)
    sys_prompt = get_sys_prompt_verifier(trace_info, add_summ_info, add_expectation_info, two_pass)
    return {"eval_prompt": eval_prompt, "sys_prompt": sys_prompt}
