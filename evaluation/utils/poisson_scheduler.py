import numpy as np
import time
import threading
import queue
import logging
import concurrent.futures
import traceback
from typing import Callable, List, Any, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm

from openhands.core.logger import openhands_logger as logger
from evaluation.utils.shared import (
    EvalMetadata, 
    EvalOutput, 
    EvalTimeoutException,
    update_progress,
    cleanup,
    is_fatal_runtime_error
)

class ThreadSafeTimeoutWrapper:
    """A thread-safe timeout wrapper for functions that can't use signal-based timeouts."""
    
    @staticmethod
    def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=None):
        """
        Run the given function with a timeout.
        
        Args:
            func: The function to run
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            timeout_seconds: Timeout in seconds, or None for no timeout
            
        Returns:
            The result of the function
            
        Raises:
            EvalTimeoutException: If the function times out
            Exception: Any exception raised by the function
        """
        if kwargs is None:
            kwargs = {}
            
        if timeout_seconds is None:
            # If no timeout specified, just run the function
            return func(*args, **kwargs)
        
        # Use ThreadPoolExecutor for timeout management
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise EvalTimeoutException(f'Function timed out after {timeout_seconds} seconds')

def _process_instance_wrapper_thread_safe(
    process_instance_func: Callable[[pd.Series, EvalMetadata, bool], EvalOutput],
    instance: pd.Series,
    metadata: EvalMetadata,
    use_mp: bool,
    max_retries: int = 5,
    timeout_seconds: Optional[int] = None,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Thread-safe wrapper for processing instances with timeout support."""
    for attempt in range(max_retries + 1):
        try:
            kwargs = {}
            # Check if process_instance_func accepts runtime_failure_count parameter
            if runtime_failure_count > 0:
                kwargs['runtime_failure_count'] = runtime_failure_count
            
            # Run with thread-safe timeout instead of using the signal-based timeout
            # which only works in the main thread
            if timeout_seconds is not None:
                result = ThreadSafeTimeoutWrapper.run_with_timeout(
                    func=process_instance_func,
                    args=(instance, metadata, use_mp),
                    kwargs=kwargs,
                    timeout_seconds=timeout_seconds
                )
            else:
                result = process_instance_func(instance, metadata, use_mp, **kwargs)
                
            return result
            
        except EvalTimeoutException as e:
            error = f'Timeout after {timeout_seconds} seconds'
            logger.exception(e)
            return EvalOutput(
                instance_id=instance.instance_id,
                test_result={},
                error=error,
            )
            
        except Exception as e:
            # Handle retries, similar to the original _process_instance_wrapper
            if attempt == max_retries:
                logger.exception(e)
                raise RuntimeError(
                    f'Maximum error retries reached for instance {instance.instance_id}'
                ) from e
                
            # Check for fatal runtime errors to increment runtime_failure_count
            error_str = type(e).__name__ + ': ' + str(e)
            if is_fatal_runtime_error(error_str):
                runtime_failure_count += 1
                logger.error(f'Runtime error detected for instance {instance.instance_id}, runtime failure count: {runtime_failure_count}')
                
            logger.error(f'Error processing instance {instance.instance_id}: {str(e)}. Retrying... (attempt {attempt + 1} of {max_retries})')
            time.sleep(5)

def run_evaluation_poisson(
    dataset: pd.DataFrame,
    metadata: EvalMetadata | None,
    output_file: str,
    rate_per_minute: float,  # Average number of tasks to launch per minute
    process_instance_func: Callable[[pd.Series, EvalMetadata, bool], EvalOutput],
    max_retries: int = 5,  # number of retries for each instance
    timeout_seconds: Optional[int] = None,
    max_concurrent_tasks: Optional[int] = None,  # Maximum number of concurrent tasks
):
    """
    Run evaluation with tasks launched according to a Poisson time distribution.
    Uses a thread-safe timeout mechanism instead of signal-based timeout.
    
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
    
    def process_task(instance, runtime_failure_count=0):
        # Acquire semaphore to limit concurrency if needed
        with concurrency_semaphore:
            try:
                # Use thread-safe wrapper instead of signal-based timeout
                result = _process_instance_wrapper_thread_safe(
                    process_instance_func=process_instance_func,
                    instance=instance,
                    metadata=metadata,
                    use_mp=False,
                    max_retries=max_retries,
                    timeout_seconds=timeout_seconds,
                    runtime_failure_count=runtime_failure_count
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
        result_thread.join(timeout=600)  # Give it ten minutes to finish processing results
        
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received. Cleaning up...\n')
        cleanup()
    finally:
        output_fp.close()
        logger.info('\nEvaluation finished.\n')
