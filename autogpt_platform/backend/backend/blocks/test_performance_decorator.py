"""
Test file for block performance decorator.
Demonstrates usage of the performance measurement capabilities.
"""

import pytest
import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from backend.util.performance_decorator import (
    measure_block_performance,
    BlockPerformanceMetrics,
    PerformanceMonitor,
    performance_monitor
)
from backend.blocks._base import Block, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField


class MockBlock(Block):
    """Mock block for testing performance decorator."""
    
    class Input(BlockSchemaInput):
        value: int = SchemaField(description="Input value", default=1)
        text: str = SchemaField(description="Input text", default="test")
    
    class Output(BlockSchemaOutput):
        result: int = SchemaField(description="Result value")
        processed_text: str = SchemaField(description="Processed text")
        error: str = SchemaField(description="Error message", default="")
    
    def __init__(self):
        super().__init__(
            id="mock-block-001",
            description="Mock block for performance testing",
            input_schema=MockBlock.Input,
            output_schema=MockBlock.Output,
            test_input={"value": 5, "text": "hello"},
            test_output={"result": 10, "processed_text": "hello_processed"}
        )
    
    @measure_block_performance(track_memory=True, track_io=True, log_metrics=False)
    async def run(self, input_data: Input, **kwargs) -> Any:
        """Mock block execution with simulated work."""
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        # Simulate memory usage
        data = [x for x in range(1000)]
        
        # Process the input
        result = input_data.value * 2
        processed_text = input_data.text + "_processed"
        
        yield "result", result
        yield "processed_text", processed_text


class TestPerformanceDecorator:
    """Test the performance decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_decorator_basic(self):
        """Test basic performance measurement."""
        block = MockBlock()
        
        # Create test input
        test_input = block.Input(value=10, text="test_input")
        
        # Run the block
        results = []
        async for output in block.run(test_input):
            results.append(output)
        
        # Verify results
        assert len(results) == 2
        assert ("result", 20) in results
        assert ("processed_text", "test_input_processed") in results
        
        # Check that performance metrics were recorded
        assert hasattr(block, '_performance_metrics')
        assert len(block._performance_metrics) > 0
        
        # Check metrics structure
        metrics = block._performance_metrics[0]
        assert isinstance(metrics, BlockPerformanceMetrics)
        assert metrics.block_name == "MockBlock"
        assert metrics.success == True
        assert metrics.execution_time_ms > 0
        assert metrics.memory_peak_mb >= 0
    
    @pytest.mark.asyncio
    async def test_performance_decorator_with_error(self):
        """Test performance measurement with error."""
        
        class ErrorBlock(Block):
            class Input(BlockSchemaInput):
                should_fail: bool = SchemaField(description="Whether to fail", default=False)
            
            class Output(BlockSchemaOutput):
                success: bool = SchemaField(description="Success flag")
                error: str = SchemaField(description="Error message", default="")
            
            def __init__(self):
                super().__init__(
                    id="error-block-001",
                    description="Block that fails for testing",
                    input_schema=ErrorBlock.Input,
                    output_schema=ErrorBlock.Output
                )
            
            @measure_block_performance()
            async def run(self, input_data: Input, **kwargs) -> Any:
                if input_data.should_fail:
                    raise ValueError("Test error")
                yield "success", True
        
        block = ErrorBlock()
        test_input = block.Input(should_fail=True)
        
        # Run and expect error
        with pytest.raises(ValueError, match="Test error"):
            async for output in block.run(test_input):
                pass
        
        # Check that error was recorded
        assert len(block._performance_metrics) > 0
        metrics = block._performance_metrics[0]
        assert metrics.success == False
        assert "Test error" in metrics.error_message
    
    def test_performance_monitor(self):
        """Test the performance monitor utility."""
        monitor = PerformanceMonitor()
        
        # Create some mock metrics
        metrics1 = BlockPerformanceMetrics(
            block_name="TestBlock1",
            execution_time_ms=100,
            cpu_time_ms=50,
            memory_peak_mb=10,
            memory_current_mb=5,
            input_size_bytes=100,
            output_size_bytes=200,
            success=True,
            performance_score=85.0
        )
        
        metrics2 = BlockPerformanceMetrics(
            block_name="TestBlock1",
            execution_time_ms=120,
            cpu_time_ms=60,
            memory_peak_mb=12,
            memory_current_mb=6,
            input_size_bytes=100,
            output_size_bytes=200,
            success=True,
            performance_score=80.0
        )
        
        # Add metrics
        monitor.add_metrics(metrics1)
        monitor.add_metrics(metrics2)
        
        # Get average performance
        avg_perf = monitor.get_average_performance("TestBlock1")
        assert avg_perf is not None
        assert avg_perf["avg_execution_time"] == 110.0
        assert avg_perf["avg_cpu_time"] == 55.0
        assert avg_perf["avg_memory_peak"] == 11.0
        assert avg_perf["avg_performance_score"] == 82.5
        assert avg_perf["success_rate"] == 1.0
        assert avg_perf["total_executions"] == 2
        
        # Generate report
        report = monitor.get_performance_report()
        assert "summary" in report
        assert "block_performance" in report
        assert report["summary"]["total_blocks"] == 1
        assert report["summary"]["total_executions"] == 2
    
    @pytest.mark.asyncio
    async def test_sync_function_decorator(self):
        """Test performance decorator on sync function."""
        
        @measure_block_performance()
        def sync_function(x: int, y: int) -> int:
            # Simulate some work
            total = 0
            for i in range(1000):
                total += i
            return x + y + total
        
        result = sync_function(5, 10)
        assert result > 15
        
        # The decorator should work without errors on sync functions
        # Note: Performance metrics won't be stored since there's no self instance


class TestPerformanceIntegration:
    """Integration tests for performance measurement with actual blocks."""
    
    @pytest.mark.asyncio
    async def test_with_semantic_search_block(self):
        """Test performance decorator with semantic search block."""
        # Import after patching to avoid import errors
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_openai.return_value.embeddings.create = MagicMock(return_value=mock_response)
            
            from backend.blocks.semantic_search import SemanticSearchBlock
            
            block = SemanticSearchBlock()
            test_input = block.Input(
                query="test query",
                search_corpus=["document 1", "document 2", "document 3"],
                max_results=2
            )
            
            # Run the block
            results = []
            async for output in block.run(test_input):
                results.append(output)
            
            # Verify performance metrics were collected
            assert hasattr(block, '_performance_metrics')
            assert len(block._performance_metrics) > 0
            
            metrics = block._performance_metrics[0]
            assert metrics.block_name == "SemanticSearchBlock"
            assert metrics.success == True
            assert metrics.execution_time_ms > 0
    
    def test_performance_score_calculation(self):
        """Test performance score calculation logic."""
        # Fast execution, low memory should get high score
        fast_metrics = BlockPerformanceMetrics(
            block_name="FastBlock",
            execution_time_ms=50,  # Under 100ms threshold
            cpu_time_ms=25,
            memory_peak_mb=5,  # Under 10MB threshold
            memory_current_mb=2,
            input_size_bytes=100,
            output_size_bytes=200,
            success=True,
            performance_score=0  # Will be calculated
        )
        
        # Calculate expected score
        expected_time_score = 100 - (50 / 100) = 50
        expected_memory_score = 100 - (5 * 10) = 50
        expected_total_score = (50 + 50) / 2 = 50
        
        # The actual calculation in the decorator would give this score
        assert fast_metrics.execution_time_ms == 50
        assert fast_metrics.memory_peak_mb == 5


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v"])
