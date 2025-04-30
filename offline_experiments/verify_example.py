# fmt: off

from llms.llm_utils import call_llm, visualize_prompt
from llms.prompt_utils import get_messages
from offline_experiments.prompts import *
from offline_experiments.vwa_specific import get_intent_message, get_prompt_constructor
from utils.string_utils import safe_format
from vwa_utils.extract_trajectory_html import extract_trajectory_data

# ===============================================
# LINK: Prompt parts
# ===============================================
# We define some template parts to be filled for each experiment case.

# The basic parts for the verifier are:
# - a system prompt, an evaluation criteria, and a response format for the evaluation (where we also specify CoT).
# - in our case, we also have a prompt to retrieve knowledge and inject it before the verification.

# For example, during verification, the LLM will receive:
# - input: [sys_prompt] [the task information] [images/video, text] (i.e., the trajectory to verify correctness) <retrieved_knowledge> [eval_criteria] [response_format]
# - output: LLM generation that must abide to the `response_format`

# During knowledge retrieval, the LLM receives:
# - input: [sys_prompt] [the task information] [k_retrieval_prompt]
# - output: the LLM generation


# --- System prompt ---
# This is the base system prompt - VWA specific
sys_prompt_base = f"""You are an intelligent agent tasked with supervising an agent navigating a web browser to accomplish a web-based task. Your job is to evaluate the agent's work, and provide feedback so it can progress towards the objective.
\n
## Here's the information you'll have:
### Objective: This is the task the agent is trying to complete.
### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.{{trace_info}}{{summary_info}}{{web_knowledge_info}}
\n
## agent's capabilities: To effectively analyze the agent's work, consider the actions it can perform. These actions fall into the following categories:
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
1. You must not not assume the agent's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.{{web_knowledge_rule}}
"""
# Note the above prompt has {} placeholders. These are filled according to what's given to the LLM in each experiment case.

trace_info_noutt = (
    "\n### The execution trace: This is a sequence of webpage screenshots, detailing the web navigation so far."
)
trace_info_utt = "\n### The execution trace: This is a sequence of webpage screenshots paired with the agent's responses, detailing the web navigation so far."
knowledge_info = "\n### General web knowledge: This is a general description of how tasks like this are typically accomplished on the web."

knowledge_rule = (
    "\n3. Use the General web knowledge as a guide, but also consider the context of the specific task given to you."
)

# For example, if we are giving the retrieved knowledge, we create the final sys_prompt as follows:
sys_prompt = safe_format(sys_prompt_base, web_knowledge_rule=knowledge_rule)
sys_prompt_noK = safe_format(sys_prompt_base, web_knowledge_rule="")
# This will fill the above prompt with a third rule at the end. Print it out to see.
print(sys_prompt)

# ---- Evaluation Criteria ----
# These are the evaluation criteria for each experiment case.

eval_criteria_ternary = """Now please provide your response.
\n
## Here is the evaluation criteria:
SUCCESS: The agent executed **all** that's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The agent executed **part of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The agent executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed."""

eval_criteria_binary = """Now please provide your response.
\n
## Here is the evaluation criteria:
SUCCESS: The agent executed **all** that's necessary to complete the objective. The task is fully accomplished.
FAILURE: The agent executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed."""

eval_criteria = {
    "ternary": eval_criteria_ternary,
    "binary": eval_criteria_binary,
}

# ---- Response format ----
# This is the final query given to the LLM to generate the response.

response_format_noCoT = """## Provide your response as follows:
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the agent can progress towards the objective]"""


response_format_basicCoT = """## Provide your response as follows:
REASONING: [Your reasoning to come up with your evaluation and feedback]
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the agent can progress towards the objective]"""


# This is the prompt to elicit the LLM to retrieve knowledge.
k_retrieval = """Please first provide the following: 
[Description of how tasks such as this are typically accomplished on the web.]"""

# This is the prompt to inject the retrieved knowledge
k_injection = """## General web knowledge:
{k}"""

# ===============================================
# 1 - Retrieve the trajectory data
# ===============================================
domain, task_id = "shopping", 368
html_path = f"./experiments/gpt-4o-2024-08-06/base_stateaware/{domain}/htmls/render_{task_id}.html"

intent_data, trajectory, meta_data = extract_trajectory_data(html_path)

# ====================================================
# 2 - Build prompt parts that are common for all cases
# ====================================================
# For VWA, these are: [the task information]; the trajectory: [images, text]


msg_intent = get_intent_message(
    intent=intent_data,
    trajectory=trajectory,
    task_id=task_id,
    add_state_idxs=[0],
    state_img_intros=["Description: Initial webpage screenshot"],
)

msg_intent_no_state = get_intent_message(
    intent=intent_data,
    trajectory=trajectory,
    task_id=task_id,
    add_state_idxs=[],
    state_img_intros=[],
)

prompt_constructor = get_prompt_constructor()


# ===============================================
# 3 - Retrieve the knowledge
# ===============================================

prompt_constructor.instruction["meta_data"]["use_agent_utterance"] = True
prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = True

interaction_history_msgs = prompt_constructor.build_interaction_history(
    trajectory, meta_data, idxs_history=list(range(len(trajectory.states)))
)


prompt_constructor.instruction["meta_data"]["use_agent_utterance"] = False
prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = False
trace_no_agent = prompt_constructor.build_interaction_history(trajectory, meta_data, idxs_history=idxs_history)
visualize_prompt(trace_no_agent, "./vis.html")


prompt = get_messages(
    inputs=[msg_intent, k_retrieval],
    sys_prompt=sys_prompt,
)
visualize_prompt(prompt, "./vis.html")


gen_config = {
    "model": "gemini-2.0-flash-001",
    "temperature": 1.0,
    "max_tokens": 8192,
    "top_p": 0.5,
    "top_k": 40,
    "response_format": None,
    "seed": 0,
    "reasoning_effort": "high",
}


_, model_messages = call_llm(
    gen_config,
    prompt,
)
k_retrieval_resp_uncond = model_messages[0].text()


k = """## General web knowledge:
A typical web task like this is accomplished by:
1.  **Search/Browse:** The user either searches for a specific item using a search bar or browses through categories and subcategories to find the desired product.
2.  **Product Selection:** Once the user finds the desired product, they click on it to view the product details page.
3.  **Product Details:** The product details page provides information such as product name, description, price, availability, customer reviews, and images.
4.  **Options/Customization:** If the product has options such as color, size, or other specifications, the user selects the desired options.
5.  **Add to Cart:** The user adds the product to their shopping cart by clicking an "Add to Cart" button.
6.  **View Cart:** The user can view the contents of their shopping cart to review the selected items and quantities.
7.  **Checkout:** The user proceeds to the checkout process.
8.  **Shipping Information:** The user provides shipping address and selects a shipping method.
9.  **Payment Information:** The user provides payment information, such as credit card details or uses a payment service like PayPal.
"""


gen_config = {
    "model": "gemini-2.0-flash-001",
    "temperature": 1,
    "top_p": None,
    "top_k": None,
    "max_tokens": 8192*20,
    "seed": None,
    "reasoning_effort": "high",
    "response_format": None,
}

neg_k_retrieval = """Please first provide the following: 
[Common pitfalls and what could be overlookedwhen trying to accomplish tasks such as this on the web.]"""


neg_k_retrieval = """Please first provide the following: 
[What would specifically look for to identify that a web task such as this is accomplished?]"""

neg_k_retrieval = """Please first provide the following: 
[How would a bad quality execution look like for a web task such as this?]"""


sys_prompt_negK = safe_format(sys_prompt_base, 
web_knowledge_rule="3. Emphasize task completion. We are less interested in efficiency for now.")

prompt = get_messages(
    inputs=[msg_intent, neg_k_retrieval],
    sys_prompt=sys_prompt_negK,
)
_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())

criteria = model_messages[0].text()

# ===============================================
# 4 - Build the prompt
# ===============================================



msg_intent_no_state = get_intent_message(
    intent=intent_data,
    trajectory=trajectory,
    task_id=task_id,
    add_state_idxs=[],
    state_img_intros=[],
)


sys_prompt = safe_format(
    sys_prompt_base,
    trace_info=trace_info_utt,
    summary_info="",
    web_knowledge_rule=knowledge_rule,
    web_knowledge_info=knowledge_info,
    privileged_hint="",
)


idxs_history = list(range(len(trajectory.states)))

prompt_constructor.instruction["meta_data"]["use_agent_utterance"] = False
prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = True
prompt_constructor.instruction["intro_execution_history"] = "## Here is the execution trace:"

interaction_history_msgs = prompt_constructor.build_interaction_history(trajectory, meta_data, idxs_history=idxs_history)


eval_criteria = """Now please provide your response.
\n
## Here is the evaluation criteria:
SUCCESS: The agent executed **all the steps** necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The agent executed **part of the steps** necessary to complete the objective. The task is partially accomplished.
FAILURE: The agent executed **mostly incorrect steps**. The task is not accomplished, and major revisions are needed."""

# eval_criteria = """Now please provide your response.
# \n
# ## Here is the evaluation criteria:
# SUCCESS: The agent executed **all** that's necessary to complete the objective. The task is fully accomplished.
# FAILURE: The agent executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed."""



# gen_config["model"] = "gemini-2.0-flash-001"
# gen_config["model"] = "gemini-2.5-flash-preview-04-17"

# gen_config["quant_bits"] = "8"

# COMPARISON: [Compare the agent's work with the general web knowledge. Make a step by step comparison of what the agent did and what is expected.]
# RELEVANCE: [Which of the steps missing are relevant in this context. If the steps are not relevant, explain why.]
# CONCLUSION: [Based on the agent's work, comparison to general web knowledge, and the objective, what are your conclusions?]


# response_format = """## Provide your response as follows:
# REASONING: [Your reasoning to come up with your evaluation and feedback]
# EVALUATION: [Your evaluation following the evaluation criteria]
# FEEDBACK: [Feedback so the agent can progress towards the objective]"""

# response_format = """## Provide your response as follows:
# EXECUTION TRACE DESCRIPTION: [Understand and describe the agent's work]
# EVALUATION: [Your evaluation following the evaluation criteria]
# FEEDBACK: [Feedback so the agent can progress towards the objective]"""



# response_format = """## Provide your response as follows:
# EXECUTION TRACE DESCRIPTION: [Understand and describe the agent's work]
# MISSING STEPS: [Which steps were not completed? Explain why they are missing.]
# ACCOMPLISHED STEPS: [Which steps were completed? Explain why they are accomplished.]
# CONCLUSION: [Based on the agent's work, comparison to general web knowledge, and the objective, what are your conclusions?]
# EVALUATION: [Your evaluation following the evaluation criteria]
# FEEDBACK: [Feedback so the agent can progress towards the objective]"""


response_format = """## Provide your response as follows:
EXECUTION TRACE DESCRIPTION: [Understand and describe the agent's work]
COMPARISON: [Compare the agent's work with the general web knowledge. Make a step by step comparison of what the agent did and what is expected.]
RELEVANCE: [Which of the steps missing are relevant in this context. If the steps are not relevant, explain why.]
CONCLUSION: [Based on the analysis, what are your conclusions?]
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the agent can progress towards the objective]"""



response_format = """## Provide your response as follows:
EXECUTION TRACE DESCRIPTION: [Understand and describe the agent's work]
COMPARISON: [Compare the agent's work with the general web knowledge. Make a step by step comparison of what the agent did and what is expected.]
ANALYSIS: [Provide a step by step analysis connecting the dots between the objective, the missing and completed steps.]
RELEVANCE: [Which of the steps missing are relevant in this context. If the steps are not relevant, explain why.]
CONCLUSION: [Based on the analysis, what are your conclusions?]
EVALUATION: [SUCCESS | PARTIAL SUCCESS | FAILURE]
FEEDBACK: [Feedback so the agent can progress towards the objective]"""


msg_intent_no_state_ = "# OBJECTIVE: Buy the phone with the skateboarder on the display image."
# msg_intent_no_state_ = "# OBJECTIVE: Buy the phone with the skateboarder on the display image (in any color)."
interaction_history_msgs[-1].contents[0].data = 'stop [The phone with the skateboarder on the display image has been added to the cart successfully.]'

gen_config["temperature"] = 0.3
prompt = get_messages(inputs=[interaction_history_msgs, msg_intent_no_state_, k, eval_criteria, response_format], sys_prompt=sys_prompt); visualize_prompt(prompt, "./vis.html")
_, model_messages = call_llm(gen_config, prompt, conversation_dir="./rascunho", call_id = task_id)
print(model_messages[0].text())






neg_example = model_messages[0].text()
k_new = f"""# General Web Knowledge:\n{criteria}
"""

prompt = get_messages(inputs=[interaction_history_msgs, msg_intent_no_state_, k_new, [eval_criteria, response_format], ], sys_prompt=sys_prompt)
visualize_prompt(prompt, "./vis.html")
_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())


gen_config["temperature"] = 1
prompt_constructor.instruction["meta_data"]["use_agent_utterance"] = False
prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = False
trace_no_agent = prompt_constructor.build_interaction_history(trajectory, meta_data, idxs_history=idxs_history)
prompt = get_messages(inputs=[trace_no_agent, msg_intent_no_state_, k, [eval_criteria, response_format], ], sys_prompt=sys_prompt)
visualize_prompt(prompt, "./vis.html")
_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())

#===============================================

sys_prompt_base = f"""You are an intelligent agent tasked with supervising an agent navigating a web browser to accomplish a web-based task. Your job is to evaluate the agent's work, and provide feedback so it can progress towards the objective.
\n
## Here's the information you'll have:
### Objective: This is the task the agent is trying to complete.
### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
### Execution trace: This is a sequence of webpage screenshots, detailing the web navigation.
### Good Example: This is a general description of how tasks like this are typically accomplished on the web.
### Bad Example: This is an idea of how a bad quality execution looks like for a web task like this.
\n
## agent's capabilities: To effectively analyze the agent's work, consider the actions it can perform. These actions fall into the following categories:
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
1. You must not not assume the agent's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.
3. Use the General web knowledge and negative example as a guide, but also consider the context of the specific task given to you.
"""
bad_example = f"""## Bad example:
{neg_example}"""

k_pos = """## Good example:
A good quality execution for a web task like this may involve:
1.  **Search/Browse:** The user either searches for a specific item using a search bar or browses through categories and subcategories to find the desired product.
2.  **Product Selection:** Once the user finds the desired product, they click on it to view the product details page.
3.  **Product Details:** The product details page provides information such as product name, description, price, availability, customer reviews, and images.
4.  **Options/Customization:** If the product has options such as color, size, or other specifications, the user selects the desired options.
5.  **Add to Cart:** The user adds the product to their shopping cart by clicking an "Add to Cart" button.
6.  **View Cart:** The user can view the contents of their shopping cart to review the selected items and quantities.
7.  **Checkout:** The user proceeds to the checkout process.
8.  **Shipping Information:** The user provides shipping address and selects a shipping method.
9.  **Payment Information:** The user provides payment information, such as credit card details or uses a payment service like PayPal.
"""

k_neg = """## Bad example:
A bad quality execution for a web task like this may involve:
1.  **Incorrect Identification:** The agent fails to correctly identify the phone with the skateboarder on the display.
2.  **Clicking on Irrelevant Elements:** The agent clicks on elements that are not related to selecting the phone or adding it to the cart (e.g., clicking on "Home," irrelevant shopping options, or compare products).
3.  **Incorrect Item Added to Cart:** Even if the agent adds an item to the cart, it's the wrong one.
4.  **Failure to Add to Cart:** The agent identifies the correct phone but fails to execute the "add to cart" action or gets stuck on the product page without adding it.
5.  **Giving Up Too Early:** The agent declares the task impossible without properly exploring the page or attempting the obvious actions.
6. **Not considering all the information:** The agent ignores the images of the website and crops provided, and does not use OCR to identify the correct elements on the webpage
"""
response_format = """## Please first provide the following:
EXECUTION TRACE DESCRIPTION: [A step by step description of the agent's work.]
COMPARISON-GOOD: [A step by step comparison of the agent's work with the good example.]
COMPARISON-BAD: [A step by step comparison of the agent's work with the bad example.]
ANALYSIS: [Analysis of the agent's work, connecting the dots between the objective, comparison to the good and bad examples, and the execution trace.]
"""

obj_text = f"""## Here is the task the agent is trying to complete:
## OBJECTIVE: **Buy** the phone with the skateboarder on the display image (in any color)."""
prompt = get_messages(inputs=[obj_text, interaction_history_msgs, obj_text, [k_pos, k_neg], obj_text, response_format], sys_prompt=sys_prompt_base); visualize_prompt(prompt, "./vis.html")

_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())
analysis = model_messages[0].text()

response_format_final = """## Now please provide your response:
## Here is the evaluation criteria:
SUCCESS: The agent executed **ALL the steps** necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The agent executed **MOST of the steps** necessary to complete the objective. The task is partially accomplished.
FAILURE: The agent executed **FEW of the steps** necessary to complete the objective. The task is not accomplished, and major revisions are needed.

## Provide your response as follows:    
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the agent can progress towards the objective]"""

prompt = get_messages(inputs=[analysis, obj_text, interaction_history_msgs, msg_intent_no_state, [k_neg, k_pos], analysis, obj_text, response_format_final], sys_prompt=sys_prompt_base); visualize_prompt(prompt, "./vis.html")

_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())


#=======================================================================================================================

#=======================================================================================================================

# 1) Describe the navigation, only images.

q = """Please provide the following:
EXECUTION TRACE DESCRIPTION: [A careful description of the navigation based on the screenshots.]"""

sys_prompt_no_agent = """You are an intelligent and helpful agent.
You will be given a sequence of screenshots representing a navigation on the web.
Your job is to provide with much detail as possible a description of the navigation.
\n
## Here's the information you'll have:
### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
\n
## To be successful, it is very important to follow the following rules:
1. Carefully describe the navigation based on the screenshots.
2. Include all the details of the navigation; do not save on any details.
"""
# prompt = get_messages(inputs=[msg_intent_no_state, trace_no_agent, q], sys_prompt=sys_prompt_no_agent)
prompt = get_messages(inputs=[trace_no_agent, q], sys_prompt=sys_prompt_no_agent)
visualize_prompt(prompt, "./vis.html")


_, model_messages = call_llm(gen_config, prompt)
exec_trace_description = model_messages[0].text()



prompt_constructor.instruction["meta_data"]["use_agent_utterance"] = True
prompt_constructor.instruction["meta_data"]["use_low_level_actions_env_parsed"] = False
trace_no_actions = prompt_constructor.build_interaction_history(trajectory, meta_data, idxs_history=idxs_history)

exec_trace_description = f"""## Here is a textual description of the execution trace:
{exec_trace_description}"""

prompt = get_messages(inputs=[msg_intent_no_state, trace_no_actions, exec_trace_description, [eval_criteria, response_format]], sys_prompt=sys_prompt_noK)


visualize_prompt(prompt, "./vis.html")
_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())



response_format = """## Provide your response as follows:
MISSING STEPS: [Which steps were not completed? Explain why they are missing.]
ACCOMPLISHED STEPS: [Which steps were completed? Explain why they are accomplished.]
ANALYSIS: [Provide a step by step analysis connecting the dots between the objective, the missing and completed steps.]
CONCLUSION: [Based on the analysis, what are your conclusions?]
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the agent can progress towards the objective]"""

prompt = get_messages(inputs=[
    trace_no_actions, exec_trace_description, 
    [f"## Here is the task to be accomplished:\n{msg_intent_no_state[0].text()}"], 
    [eval_criteria, response_format]], sys_prompt=sys_prompt_noK)
visualize_prompt(prompt, "./vis.html")


prompt = get_messages(inputs=[
    trace_no_agent, #exec_trace_description, 
    [f"## Here is the task to be accomplished:\n{msg_intent_no_state[0].text()}"], 
    [eval_criteria, response_format]], sys_prompt=sys_prompt_noK)
visualize_prompt(prompt, "./vis.html")

_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())


#===============================================

response_format = """## Provide your response as follows:
REASONING: [Your reasoning to come up with your evaluation and feedback]
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the agent can progress towards the objective]"""


gen_config["temperature"] = 1
prompt = get_messages(inputs=[
    trace_no_agent,
    [f"## Here is the task to be accomplished:\n{intent_data['text']}"], 
    [eval_criteria, response_format]], sys_prompt=sys_prompt_noK)


sys_prompt = f"""You are an intelligent agent tasked with supervising an agent navigating a web browser to accomplish a web-based task. Your job is to evaluate the agent's work, and provide feedback so it can progress towards the objective.
\n
## Here's the information you'll have:
### Objective: This is the task the agent is trying to complete.
### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.{{trace_info}}{{summary_info}}{{web_knowledge_info}}
\n
## To be successful, it is very important to follow the following rules:
1. You must not not assume the agent's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.{{web_knowledge_rule}}
"""

intent_data['text']
intent_ = 'Buy the phone with the skateboarder on the display image.'
q = f"""Based on the sequences of screenshots detailing the navigation, and considering the objective as:
## OBJECTIVE:{intent_}.
\n
What do you consider more accurate?
SUCCESS: The agent executed **all the steps** necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The agent executed **part of the steps** necessary to complete the objective. The task is partially accomplished.
FAILURE: The agent executed **mostly incorrect steps**. The task is not accomplished, and major revisions are needed."""


trace_no_agent_modif = trace_no_agent[0:4] + [trace_no_agent[-2]]
prompt = get_messages(inputs=[trace_no_agent_modif, q], sys_prompt=sys_prompt)
    # [eval_criteria, response_format]], sys_prompt=sys_prompt)

visualize_prompt(prompt, "./vis.html")

_, model_messages = call_llm(gen_config, prompt)
print(model_messages[0].text())
