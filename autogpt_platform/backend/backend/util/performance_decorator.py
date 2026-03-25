"""
Block Performance Decorator for AutoGPT

Provides comprehensive performance measurement capabilities for blocks including:
- Execution time tracking
- CPU usage monitoring
- Memory usage tracking
- Input/output size measurement
- Performance scoring
- Metrics logging and collection
"""

import inspect
import json
import logging
import os
import time
import tracemalloc
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__))


class BlockPerformanceMetrics(BaseModel):
    """Performance metrics for block execution."""
    block_name: str
    execution_time_ms: float
    cpu_time_ms: float
    memory_peak_mb: float
    memory_current_mb: float
    input_size_bytes: int
    output_size_bytes: int
    success: bool
    error_message: Optional[str] = None
    performance_score: float  # 0-100, higher is better


def measure_block_performance(
    track_memory: bool = True,
    track_io: bool = True,
    log_metrics: bool = True,
) -> Callable:
    """
    Decorator to measure comprehensive performance metrics for blocks.
    
    Args:
        track_memory: Whether to track memory usage
        track_io: Whether to track input/output sizes
        log_metrics: Whether to log metrics after execution
    
    Usage:
        @measure_block_performance(track_memory=True, track_io=True)
        async def my_block_run(self, input_data, **kwargs):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Start measurements
            start_time = time.time()
            start_cpu = os.times()[0] + os.times()[1]
            
            # Start memory tracking if requested
            if track_memory:
                tracemalloc.start()
            
            # Calculate input size if requested
            input_size = 0
            if track_io and args:
                try:
                    if len(args) > 1 and hasattr(args[1], 'dict'):
                        input_size = len(json.dumps(args[1].dict()).encode())
                except Exception:
                    input_size = 0
            
            success = True
            error_message = None
            result = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # End measurements
                end_time = time.time()
                end_cpu = os.times()[0] + os.times()[1]
                execution_time_ms = (end_time - start_time) * 1000
                cpu_time_ms = (end_cpu - start_cpu) * 1000
                
                # Get memory metrics
                memory_peak_mb = 0
                memory_current_mb = 0
                if track_memory and tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    memory_current_mb = current / (1024 * 1024)
                    memory_peak_mb = peak / (1024 * 1024)
                    tracemalloc.stop()
                
                # Calculate output size
                output_size = 0
                if track_io and result is not None:
                    try:
                        if hasattr(result, '__anext__'):
                            output_size = 100  # Rough estimate for generators
                        else:
                            output_size = len(str(result).encode())
                    except:
                        output_size = 0
                
                # Calculate performance score (0-100)
                time_score = max(0, 100 - (execution_time_ms / 100))  # Penalize >100ms
                memory_score = max(0, 100 - (memory_peak_mb * 10))  # Penalize >10MB
                performance_score = (time_score + memory_score) / 2
                
                # Create metrics object
                block_name = func.__self__.__class__.__name__ if hasattr(func, '__self__') else func.__name__
                metrics = BlockPerformanceMetrics(
                    block_name=block_name,
                    execution_time_ms=execution_time_ms,
                    cpu_time_ms=cpu_time_ms,
                    memory_peak_mb=memory_peak_mb,
                    memory_current_mb=memory_current_mb,
                    input_size_bytes=input_size,
                    output_size_bytes=output_size,
                    success=success,
                    error_message=error_message,
                    performance_score=performance_score
                )
                
                # Log metrics if requested
                if log_metrics:
                    logger.info(
                        f"Block Performance - {block_name}: "
                        f"Time={execution_time_ms:.2f}ms, "
                        f"CPU={cpu_time_ms:.2f}ms, "
                        f"Memory={memory_peak_mb:.2f}MB, "
                        f"Score={performance_score:.1f}/100"
                    )
                
                # Store metrics in block instance if available
                instance = args[0]
                if hasattr(instance, '_performance_metrics'):
                    instance._performance_metrics.append(metrics)
                elif hasattr(args[0], '__self__') and hasattr(args[0].__self__, '_performance_metrics'):
                    args[0].__self__._performance_metrics.append(metrics)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            start_cpu = os.times()[0] + os.times()[1]
            
            # Start memory tracking if requested
            if track_memory:
                tracemalloc.start()
            
            input_size = 0
            if track_io and args:
                try:
                    if len(args) > 1 and hasattr(args[1], 'dict'):
                        input_size = len(json.dumps(args[1].dict()).encode())
                except Exception:
                    input_size = 0
            
            success = True
            error_message = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_cpu = os.times()[0] + os.times()[1]
                execution_time_ms = (end_time - start_time) * 1000
                cpu_time_ms = (end_cpu - start_cpu) * 1000
                
                memory_peak_mb = 0
                memory_current_mb = 0
                if track_memory and tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    memory_current_mb = current / (1024 * 1024)
                    memory_peak_mb = peak / (1024 * 1024)
                    tracemalloc.stop()
                
                output_size = 0
                if track_io and result is not None:
                    try:
                        output_size = len(str(result).encode())
                    except:
                        output_size = 0
                
                time_score = max(0, 100 - (execution_time_ms / 100))
                memory_score = max(0, 100 - (memory_peak_mb * 10))
                performance_score = (time_score + memory_score) / 2
                
                block_name = func.__self__.__class__.__name__ if hasattr(func, '__self__') else func.__name__
                metrics = BlockPerformanceMetrics(
                    block_name=block_name,
                    execution_time_ms=execution_time_ms,
                    cpu_time_ms=cpu_time_ms,
                    memory_peak_mb=memory_peak_mb,
                    memory_current_mb=memory_current_mb,
                    input_size_bytes=input_size,
                    output_size_bytes=output_size,
                    success=success,
                    error_message=error_message,
                    performance_score=performance_score
                )
                
                if log_metrics:
                    logger.info(
                        f"Block Performance - {block_name}: "
                        f"Time={execution_time_ms:.2f}ms, "
                        f"CPU={cpu_time_ms:.2f}ms, "
                        f"Memory={memory_peak_mb:.2f}MB, "
                        f"Score={performance_score:.1f}/100"
                    )
        
        # Return appropriate wrapper based on function type
        # Check for both coroutines and async generators (Block.run() methods are async generators)
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Performance monitoring utilities
class PerformanceMonitor:
    """Utility class for monitoring and analyzing block performance."""
    
    def __init__(self):
        self.metrics_history: Dict[str, list[BlockPerformanceMetrics]] = {}
    
    def add_metrics(self, metrics: BlockPerformanceMetrics) -> None:
        """Add performance metrics to the history."""
        if metrics.block_name not in self.metrics_history:
            self.metrics_history[metrics.block_name] = []
        self.metrics_history[metrics.block_name].append(metrics)
    
    def get_average_performance(self, block_name: str) -> Optional[Dict[str, float]]:
        """Get average performance metrics for a block."""
        if block_name not in self.metrics_history:
            return None
        
        metrics_list = self.metrics_history[block_name]
        if not metrics_list:
            return None
        
        return {
            "avg_execution_time": sum(m.execution_time_ms for m in metrics_list) / len(metrics_list),
            "avg_cpu_time": sum(m.cpu_time_ms for m in metrics_list) / len(metrics_list),
            "avg_memory_peak": sum(m.memory_peak_mb for m in metrics_list) / len(metrics_list),
            "avg_performance_score": sum(m.performance_score for m in metrics_list) / len(metrics_list),
            "success_rate": sum(1 for m in metrics_list if m.success) / len(metrics_list),
            "total_executions": len(metrics_list)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "summary": {
                "total_blocks": len(self.metrics_history),
                "total_executions": sum(len(metrics) for metrics in self.metrics_history.values())
            },
            "block_performance": {}
        }
        
        for block_name in self.metrics_history:
            avg_perf = self.get_average_performance(block_name)
            if avg_perf:
                report["block_performance"][block_name] = avg_perf
        
        return report


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
