import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .core import Pipeline, PipelineNode

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class BatchConfig:
    batch_size: int = 32
    max_wait_time: float = 1.0  # seconds
    min_batch_size: int = 1


class BatchNode(PipelineNode):
    """Node that processes data in batches for improved performance"""
    def __init__(
        self,
        name: str,
        func: Callable[[List[Any]], List[Any]],
        config: BatchConfig = BatchConfig()
    ):
        super().__init__(name, func, is_async=True)
        self.config = config
        self._batch: List[Any] = []
        self._batch_lock = asyncio.Lock()
        self._last_batch_time = asyncio.get_event_loop().time()
        self._batch_task: Optional[asyncio.Task] = None

    async def process(self, data: Any) -> Any:
        async with self._batch_lock:
            self._batch.append(data)
            current_time = asyncio.get_event_loop().time()

            # Process batch if it's full or enough time has passed
            if (len(self._batch) >= self.config.batch_size or
                (len(self._batch) >= self.config.min_batch_size and
                 current_time - self._last_batch_time >= self.config.max_wait_time)):
                return await self._process_batch()

            # Start a timer to process partial batch
            if not self._batch_task:
                self._batch_task = asyncio.create_task(self._batch_timer())

            return None

    async def _batch_timer(self):
        await asyncio.sleep(self.config.max_wait_time)
        if len(self._batch) >= self.config.min_batch_size:
            await self._process_batch()
        self._batch_task = None

    async def _process_batch(self) -> Any:
        if not self._batch:
            return None

        current_batch = self._batch
        self._batch = []
        self._last_batch_time = asyncio.get_event_loop().time()

        try:
            results = await self.func(current_batch)
            return results
        except Exception as e:
            logger.error(f"Error processing batch in node {self.name}: {str(e)}")
            raise


class CachedNode(PipelineNode):
    """Node with built-in caching support"""
    def __init__(
        self,
        name: str,
        func: Callable,
        is_async: bool = False,
        cache_size: int = 1000
    ):
        super().__init__(name, func, is_async)
        self.cache: Dict[Any, Any] = {}
        self.cache_size = cache_size

    async def process(self, data: Any) -> Any:
        try:
            # Convert data to hashable type if needed
            cache_key = self._get_cache_key(data)

            # Check cache
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Process data
            result = await super().process(data)

            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Cache error in node {self.name}: {str(e)}")
            # Fallback to direct processing if caching fails
            return await super().process(data)

    def _get_cache_key(self, data: Any) -> Any:
        """Convert input data to a hashable type for caching"""
        if isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, (list, tuple)):
            return tuple(map(self._get_cache_key, data))
        elif isinstance(data, dict):
            return tuple(sorted(
                (k, self._get_cache_key(v)) for k, v in data.items()
            ))
        return str(data)


class ParallelNode(PipelineNode):
    """Node that processes data in parallel using a thread pool"""
    def __init__(
        self,
        name: str,
        func: Callable,
        is_async: bool = False,
        max_workers: int = None
    ):
        super().__init__(name, func, is_async)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    async def process(self, data: Any) -> Any:
        if self.is_async:
            return await self.func(data)
        else:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(self.thread_pool, self.func, data)
            finally:
                loop.close()

    def __del__(self):
        self.thread_pool.shutdown(wait=False)


class TypedNode(PipelineNode, Generic[T, U]):
    """Type-safe node with clear input/output types"""
    def __init__(
        self,
        name: str,
        func: Callable[[T], U],
        is_async: bool = False
    ):
        super().__init__(name, func, is_async)
        self.input_type = T
        self.output_type = U

    async def process(self, data: Any) -> U:
        if not isinstance(data, self.input_type):
            raise TypeError(
                f"Node {self.name} expected input type {self.input_type}, "
                f"but got {type(data)}"
            )
        result = await super().process(data)
        if not isinstance(result, self.output_type):
            raise TypeError(
                f"Node {self.name} expected output type {self.output_type}, "
                f"but got {type(result)}"
            )
        return result


class FilterNode(PipelineNode):
    """Node that filters data based on a condition"""
    def __init__(self, name: str, condition: Callable[[Any], bool]):
        super().__init__(name, None, is_async=False)
        self.condition = condition

    async def process(self, data: Any) -> Optional[Any]:
        try:
            if self.condition(data):
                return data
            return None
        except Exception as e:
            logger.error(f"Error in filter node {self.name}: {str(e)}")
            raise


class OptimizedPipeline(Pipeline):
    """Pipeline with additional optimization features"""
    def __init__(self, name: str):
        super().__init__(name)
        self._parallel_execution = True

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        results = {}

        async def process_node(node: PipelineNode, data: Any):
            try:
                result = await node.process(data)
                if result is not None:  # Only store non-None results
                    results[node.name] = result

                # Process downstream nodes in parallel if enabled
                if self._parallel_execution and node.downstream_nodes:
                    await asyncio.gather(*(
                        process_node(downstream, result)
                        for downstream in node.downstream_nodes
                    ))
                else:
                    for downstream in node.downstream_nodes:
                        await process_node(downstream, result)

            except Exception as e:
                logger.error(f"Error executing node {node.name}: {str(e)}")
                raise

        # Start execution from start nodes
        if self._parallel_execution and len(self.start_nodes) > 1:
            await asyncio.gather(*(
                process_node(start_node, input_data)
                for start_node in self.start_nodes
            ))
        else:
            for start_node in self.start_nodes:
                await process_node(start_node, input_data)

        return results

    def enable_parallel_execution(self, enabled: bool = True):
        """Enable or disable parallel execution of nodes"""
        self._parallel_execution = enabled
