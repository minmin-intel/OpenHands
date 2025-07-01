import numpy as np
import time
import threading
import queue
import logging
from typing import Callable, List, Any, Dict, Tuple
import pandas as pd
from tqdm import tqdm

from openhands.core.logger import openhands_logger as logger
from evaluation.utils.shared import (
    EvalMetadata, 
    EvalOutput, 
    _process_instance_wrapper, 
    update_progress,
    cleanup
)

def run_evaluation_poisson(
    dataset: pd.DataFrame,
    metadata: EvalMetadata | None,
    output_file: str,
    rate_per_minute: float,  # Average number of tasks to launch per minute
    process_instance_func: Callable[[pd.Series, EvalMetadata, bool], EvalOutput],
    max_retries: int = 5,  # number of retries for each instance
    timeout_seconds: int | None = None,
    max_concurrent_tasks: int | None = None,  # Maximum number of concurrent tasks
):
    """
    Run evaluation with tasks launched according to a Poisson time distribution.
    
    Args:
        dataset: DataFrame containing instances to process
        metadata: Metadata for the evaluation
        output_file: Path to file where results will be written
        rate_per_minute: Average number of tasks to launch per minute (λ parameter for Poisson distribution)
        process_instance_func: Function to process each instance
        max_retries: Maximum number of retries for each instance
        timeout_seconds: Timeout for each instance
        max_concurrent_tasks: Maximum number of concurrent tasks (if None, unlimited)
    """
    # Use the OpenHands logger
    
    if metadata is not None:
        logger.info(
            f'Evaluation started with Agent {metadata.agent_class}:\n'
            f'model {metadata.llm_config.model}, max iterations {metadata.max_iterations}.\n'
            f'Using Poisson time distribution with rate {rate_per_minute} tasks per minute.\n'
        )
    else:
        logger.warning('Running evaluation without metadata.')
        logger.info(f'Using Poisson time distribution with rate {rate_per_minute} tasks per minute.')
    
    # Convert rate per minute to rate per second for interval calculations
    rate_per_second = rate_per_minute / 60.0
    
    # Generate Poisson distributed intervals (in seconds)
    # For a Poisson process, the time between events follows an exponential distribution
    # with parameter λ (rate_per_second)
    def generate_intervals(count):
        return np.random.exponential(scale=1.0/rate_per_second, size=count)
    
    total_instances = len(dataset)
    pbar = tqdm(total=total_instances, desc='Instances processed')
    output_fp = open(output_file, 'a')
    
    # Queue for results
    result_queue = queue.Queue()
    
    # Semaphore to limit concurrent tasks if specified
    if max_concurrent_tasks:
        concurrency_semaphore = threading.Semaphore(max_concurrent_tasks)
    else:
        # Dummy semaphore that doesn't limit anything
        concurrency_semaphore = threading.Semaphore(value=total_instances)
    
    # Track active threads
    active_threads = []
    
    def process_task(instance):
        # Acquire semaphore to limit concurrency if needed
        with concurrency_semaphore:
            try:
                result = _process_instance_wrapper(
                    process_instance_func=process_instance_func,
                    instance=instance,
                    metadata=metadata,
                    use_mp=False,
                    max_retries=max_retries,
                    timeout_seconds=timeout_seconds,
                )
                result_queue.put(result)
            except Exception as e:
                logger.exception(f"Error processing instance {instance.instance_id}: {str(e)}")
                # Put a placeholder result to keep track of progress
                result_queue.put(EvalOutput(
                    instance_id=instance.instance_id,
                    error=f"Thread exception: {str(e)}"
                ))
    
    # Thread to handle results and update progress
    def result_handler():
        completed = 0
        try:
            while completed < total_instances:
                try:
                    result = result_queue.get(timeout=1)
                    update_progress(result, pbar, output_fp)
                    completed += 1
                except queue.Empty:
                    # No result available yet, just continue waiting
                    continue
        except Exception as e:
            logger.exception(f"Error in result handler: {str(e)}")
    
    # Start result handler thread
    result_thread = threading.Thread(target=result_handler)
    result_thread.daemon = True
    result_thread.start()
    
    try:
        # Generate time intervals for launching tasks
        intervals = generate_intervals(total_instances - 1)  # -1 because first task starts immediately
        
        # Launch tasks according to Poisson distribution
        for i, (_, instance) in enumerate(dataset.iterrows()):
            # Start the task in a new thread
            thread = threading.Thread(target=process_task, args=(instance,))
            thread.daemon = True
            thread.start()
            active_threads.append(thread)
            
            # Wait for the Poisson interval before launching the next task
            # except for the last task
            if i < total_instances - 1:
                interval = intervals[i]
                logger.info(f"Waiting {interval:.2f} seconds before starting next task")
                time.sleep(interval)
        
        # Wait for all threads to complete
        for thread in active_threads:
            thread.join()
        
        # Wait for result handler to process all results
        result_thread.join(timeout=60)  # Give it a minute to finish processing results
        
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received. Cleaning up...\n')
        cleanup()
    finally:
        output_fp.close()
        logger.info('\nEvaluation finished.\n')
