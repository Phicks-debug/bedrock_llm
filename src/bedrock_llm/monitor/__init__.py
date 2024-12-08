import logging
import time
from contextvars import ContextVar
from datetime import datetime
from functools import wraps

import psutil
import pytz
from termcolor import cprint

# Define ContextVar
start_time_perf = ContextVar("start_time_perf", default=time.perf_counter())


def __get_performance_metrics(func, start_datetime, start_memory):
    end_time = time.perf_counter()
    memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory

    return {
        "function": func.__name__,
        "start_time": start_datetime,
        "duration": end_time - start_time_perf.get(),
        "memory_used": memory_used,
    }


def __print_metrics(metrics):
    cprint("\n" + "=" * 50, "blue")
    cprint("[Performance Metrics]", "blue")
    cprint(f"Function    : {metrics['function']}", "blue")
    cprint(
        f"Start Time  : {metrics['start_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "blue",
    )
    cprint(f"Duration    : {metrics['duration']:.2f} seconds", "blue")
    cprint(f"Memory Used : {metrics['memory_used']:.2f} MB", "blue")
    cprint("=" * 50, "blue")


def monitor_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        token = start_time_perf.set(time.perf_counter())  # Store performance counter
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_datetime = datetime.now(vietnam_tz)

        try:
            result = await func(*args, **kwargs)
            metrics = __get_performance_metrics(func, start_datetime, start_memory)
            __print_metrics(metrics)
            return result
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time_perf.get()
            cprint(
                f"\n[ERROR] {func.__name__} failed after {execution_time:.2f} seconds",
                "red",
            )
            cprint(f"Error: {str(e)}", "red")
            raise e
        finally:
            # Reset the context
            start_time_perf.reset(token)

    return wrapper


def monitor_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        token = start_time_perf.set(time.perf_counter())  # Store performance counter
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_datetime = datetime.now(vietnam_tz)

        try:
            result = func(*args, **kwargs)
            metrics = __get_performance_metrics(func, start_datetime, start_memory)
            __print_metrics(metrics)
            return result
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time_perf.get()
            cprint(
                f"\n[ERROR] {func.__name__} failed after {execution_time:.2f} seconds",
                "red",
            )
            cprint(f"Error: {str(e)}", "red")
            raise e
        finally:
            # Reset the context
            start_time_perf.reset(token)

    return wrapper


def __setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def log_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
        logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise e

    return wrapper


def log_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
        logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully. Result: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise e

    return wrapper


# Setup logging when the module is imported
__setup_logging()
