import asyncio
import gc

from llms.llm_utils import batch_call_llm, call_llm
from utils.logger_utils import logger

from .config_run import dump_exper_args


def run_batch_mode(all_configs, task_ids, run_config, gen_config, build_llm_call_args_fn):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    for _, config in all_configs.items():
        batch_call_llm_args = {
            "prompts": [],
            "conversation_dirs": [],
            "usage_dirs": [],
            "call_ids": [],
        }
        jobs_data = []

        dump_exper_args(config)
        for task_id in task_ids:
            if "k_config" not in config["prompt_args"]:
                continue
            jobs_data.append((task_id, config))

        # Create the prompts in parallel
        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(build_llm_call_args_fn, task_id, config, run_config): (task_id, config)
                for task_id, config in jobs_data
            }
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                prompt, conversation_dir, usage_dir, call_id = result
                if not prompt:
                    continue
                batch_call_llm_args["prompts"].append(prompt)
                batch_call_llm_args["conversation_dirs"].append(conversation_dir)
                batch_call_llm_args["usage_dirs"].append(usage_dir)
                batch_call_llm_args["call_ids"].append(call_id)

        if len(batch_call_llm_args["prompts"]) == 0:
            logger.info(f"No prompts to run for config {config}")
            continue

        logger.info(f"Running {len(batch_call_llm_args['prompts'])} calls in batch mode")
        _, _ = batch_call_llm(
            gen_kwargs=gen_config,
            prompts=batch_call_llm_args["prompts"],
            conversation_dirs=batch_call_llm_args["conversation_dirs"],
            usage_dirs=batch_call_llm_args["usage_dirs"],
            call_ids=batch_call_llm_args["call_ids"],
            max_batch_size=run_config["max_batch_size"],
            num_workers=run_config["num_processes"],
            multiprocess_mode=run_config.get("multiprocess_batch_mode", False),
            verbose=True,
            return_outputs=False,
        )
        gc.collect()
        logger.info(f"Finished config: {config}")


async def run_sequential(all_configs, task_ids, run_config, gen_config, build_llm_call_args_fn):
    # Create an asyncio Queue to hold (prompt, config, task_id) tuples.
    queue = asyncio.Queue()

    async def async_call_llm(prompt, gen_config, conversation_dir, usage_dir, call_id):
        try:
            return call_llm(
                prompt=prompt,
                gen_kwargs=gen_config,
                conversation_dir=conversation_dir,
                usage_dir=usage_dir,
                call_id=call_id,
                verbose=True,
            )
        except Exception as e:
            logger.warning(f"Error executing call associated to {conversation_dir}, call_id: {call_id}: {e}")
            return None

    async def producer():
        for _, config in all_configs.items():
            dump_exper_args(config)
            try:
                for task_id in task_ids:
                    llm_call_args = build_llm_call_args_fn(task_id, config, run_config)
                    if not llm_call_args:
                        continue
                    # Enqueue the prompt and associated call parameters.
                    await queue.put(llm_call_args)
            except Exception as e:
                logger.warning(f"Error creating prompt for task {task_id}, config {config}: {e}")

        # Signal that production is done by enqueuing a sentinel.
        await queue.put(None)

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                # No more items to process.
                break
            prompt, conversation_dir, usage_dir, call_id = item
            # Sequential asynchronous call: await ensures one at a time.
            await async_call_llm(
                prompt=prompt,
                gen_config=gen_config,
                conversation_dir=conversation_dir,
                usage_dir=usage_dir,
                call_id=call_id,
            )
            queue.task_done()

    # Start the producer task (builds prompts concurrently)
    producer_task = asyncio.create_task(producer())

    # Run the consumer which processes llm calls sequentially.
    await consumer()
    await queue.join()  # Ensures all queued items are processed.
    await producer_task
