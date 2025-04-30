## README
To get an idea of how the code for the offline verification works, check the `offline_experiments/verify_example/verify_example.ipynb`

For each environment, the most important is to to have a function that builds the data to be verified by the LLM based on previous executions from the generator (generator = another LLM, Agent, diffusion policy, oracle policy, etc.)

See below for other things needed. 

It's easy for me to work on (2) and (3) below and any other necessary changes to the code, conditional that I have an example of how the data in (1) would look like to be parsed.

For a new environment, here's what's needed :
1) A function to build the data to be verified by the LLM
    - That is, a trace of execution in `images`, `text`, and the `objective` the generator is trying to accomplish
  
2) A base system prompt containing any environment-specific relevant info

3) Possibly small changes to the first-pass prompt and evaluation criterias


With this during verification, the LLM will receive:
- input: `[sys_prompt] [the objective] [execution trace] <first_pass response> [query for verification]`
- output: LLM generation that must abide to a `response_format` defined in `[query for verification]`

And the code will run in parallel all cases for LLMs of interest.

See jupyter for an example based on the VWA environment, if it helps understanding.
- NOTE: mostly for context; you'll not be able to run the code as it has some VWA-specific stuff.
