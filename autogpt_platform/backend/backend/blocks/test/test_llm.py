from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.model import NodeExecutionStats


class TestLLMStatsTracking:
    """Test that LLM blocks correctly track token usage statistics."""

    @pytest.mark.asyncio
    async def test_llm_call_returns_token_counts(self):
        """Test that llm_call returns proper token counts in LLMResponse."""
        import backend.blocks.llm as llm

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Test response", tool_calls=None))
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        # Test with mocked OpenAI response
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            response = await llm.llm_call(
                credentials=llm.TEST_CREDENTIALS,
                llm_model=llm.LlmModel.GPT4O,
                prompt=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )

            assert isinstance(response, llm.LLMResponse)
            assert response.prompt_tokens == 10
            assert response.completion_tokens == 20
            assert response.response == "Test response"

    @pytest.mark.asyncio
    async def test_ai_structured_response_block_tracks_stats(self):
        """Test that AIStructuredResponseGeneratorBlock correctly tracks stats."""
        from unittest.mock import patch

        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()

        # Mock the llm_call method
        async def mock_llm_call(*args, **kwargs):
            return llm.LLMResponse(
                raw_response="",
                prompt=[],
                response='<json_output id="test123456">{"key1": "value1", "key2": "value2"}</json_output>',
                tool_calls=None,
                prompt_tokens=15,
                completion_tokens=25,
                reasoning=None,
            )

        block.llm_call = mock_llm_call  # type: ignore

        # Run the block
        input_data = llm.AIStructuredResponseGeneratorBlock.Input(
            prompt="Test prompt",
            expected_format={"key1": "desc1", "key2": "desc2"},
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore  # type: ignore
        )

        outputs = {}
        # Mock secrets.token_hex to return consistent ID
        with patch("secrets.token_hex", return_value="test123456"):
            async for output_name, output_data in block.run(
                input_data, credentials=llm.TEST_CREDENTIALS
            ):
                outputs[output_name] = output_data

        # Check stats
        assert block.execution_stats.input_token_count == 15
        assert block.execution_stats.output_token_count == 25
        assert block.execution_stats.llm_call_count == 1
        assert block.execution_stats.llm_retry_count == 0

        # Check output
        assert "response" in outputs
        assert outputs["response"] == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_ai_text_generator_block_tracks_stats(self):
        """Test that AITextGeneratorBlock correctly tracks stats through delegation."""
        import backend.blocks.llm as llm

        block = llm.AITextGeneratorBlock()

        # Mock the underlying structured response block
        async def mock_llm_call(input_data, credentials):
            # Simulate the structured block setting stats
            block.execution_stats = NodeExecutionStats(
                input_token_count=30,
                output_token_count=40,
                llm_call_count=1,
            )
            return "Generated text"  # AITextGeneratorBlock.llm_call returns a string

        block.llm_call = mock_llm_call  # type: ignore

        # Run the block
        input_data = llm.AITextGeneratorBlock.Input(
            prompt="Generate text",
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        outputs = {}
        async for output_name, output_data in block.run(
            input_data, credentials=llm.TEST_CREDENTIALS
        ):
            outputs[output_name] = output_data

        # Check stats
        assert block.execution_stats.input_token_count == 30
        assert block.execution_stats.output_token_count == 40
        assert block.execution_stats.llm_call_count == 1

        # Check output - AITextGeneratorBlock returns the response directly, not in a dict
        assert outputs["response"] == "Generated text"

    @pytest.mark.asyncio
    async def test_stats_accumulation_with_retries(self):
        """Test that stats correctly accumulate across retries."""
        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()

        # Counter to track calls
        call_count = 0

        async def mock_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # First call returns invalid format
            if call_count == 1:
                return llm.LLMResponse(
                    raw_response="",
                    prompt=[],
                    response='<json_output id="test123456">{"wrong": "format"}</json_output>',
                    tool_calls=None,
                    prompt_tokens=10,
                    completion_tokens=15,
                    reasoning=None,
                )
            # Second call returns correct format
            else:
                return llm.LLMResponse(
                    raw_response="",
                    prompt=[],
                    response='<json_output id="test123456">{"key1": "value1", "key2": "value2"}</json_output>',
                    tool_calls=None,
                    prompt_tokens=20,
                    completion_tokens=25,
                    reasoning=None,
                )

        block.llm_call = mock_llm_call  # type: ignore

        # Run the block with retry
        input_data = llm.AIStructuredResponseGeneratorBlock.Input(
            prompt="Test prompt",
            expected_format={"key1": "desc1", "key2": "desc2"},
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            retry=2,
        )

        outputs = {}
        # Mock secrets.token_hex to return consistent ID
        with patch("secrets.token_hex", return_value="test123456"):
            async for output_name, output_data in block.run(
                input_data, credentials=llm.TEST_CREDENTIALS
            ):
                outputs[output_name] = output_data

        # Check stats - should accumulate both calls
        # For 2 attempts: attempt 1 (failed) + attempt 2 (success) = 2 total
        # but llm_call_count is only set on success, so it shows 1 for the final successful attempt
        assert block.execution_stats.input_token_count == 30  # 10 + 20
        assert block.execution_stats.output_token_count == 40  # 15 + 25
        assert block.execution_stats.llm_call_count == 2  # retry_count + 1 = 1 + 1 = 2
        assert block.execution_stats.llm_retry_count == 1

    @pytest.mark.asyncio
    async def test_ai_text_summarizer_multiple_chunks(self):
        """Test that AITextSummarizerBlock correctly accumulates stats across multiple chunks."""
        import backend.blocks.llm as llm

        block = llm.AITextSummarizerBlock()

        # Track calls to simulate multiple chunks
        call_count = 0

        async def mock_llm_call(input_data, credentials):
            nonlocal call_count
            call_count += 1

            # Create a mock block with stats to merge from
            mock_structured_block = llm.AIStructuredResponseGeneratorBlock()
            mock_structured_block.execution_stats = NodeExecutionStats(
                input_token_count=25,
                output_token_count=15,
                llm_call_count=1,
            )

            # Simulate merge_llm_stats behavior
            block.merge_llm_stats(mock_structured_block)

            if "final_summary" in input_data.expected_format:
                return {"final_summary": "Final combined summary"}
            else:
                return {"summary": f"Summary of chunk {call_count}"}

        block.llm_call = mock_llm_call  # type: ignore

        # Create long text that will be split into chunks
        long_text = " ".join(["word"] * 1000)  # Moderate size to force ~2-3 chunks

        input_data = llm.AITextSummarizerBlock.Input(
            text=long_text,
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            max_tokens=100,  # Small chunks
            chunk_overlap=10,
        )

        # Run the block
        outputs = {}
        async for output_name, output_data in block.run(
            input_data, credentials=llm.TEST_CREDENTIALS
        ):
            outputs[output_name] = output_data

        # Block finished - now grab and assert stats
        assert block.execution_stats is not None
        assert call_count > 1  # Should have made multiple calls
        assert block.execution_stats.llm_call_count > 0
        assert block.execution_stats.input_token_count > 0
        assert block.execution_stats.output_token_count > 0

        # Check output
        assert "summary" in outputs
        assert outputs["summary"] == "Final combined summary"

    @pytest.mark.asyncio
    async def test_ai_text_summarizer_real_llm_call_stats(self):
        """Test AITextSummarizer with real LLM call mocking to verify llm_call_count."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import backend.blocks.llm as llm

        block = llm.AITextSummarizerBlock()

        # Mock the actual LLM call instead of the llm_call method
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_response = MagicMock()
            # Return different responses for chunk summary vs final summary
            if call_count == 1:
                mock_response.choices = [
                    MagicMock(
                        message=MagicMock(
                            content='<json_output id="test123456">{"summary": "Test chunk summary"}</json_output>',
                            tool_calls=None,
                        )
                    )
                ]
            else:
                mock_response.choices = [
                    MagicMock(
                        message=MagicMock(
                            content='<json_output id="test123456">{"final_summary": "Test final summary"}</json_output>',
                            tool_calls=None,
                        )
                    )
                ]
            mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=30)
            return mock_response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = mock_create

            # Test with very short text (should only need 1 chunk + 1 final summary)
            input_data = llm.AITextSummarizerBlock.Input(
                text="This is a short text.",
                model=llm.LlmModel.GPT4O,
                credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
                max_tokens=1000,  # Large enough to avoid chunking
            )

            # Mock secrets.token_hex to return consistent ID
            with patch("secrets.token_hex", return_value="test123456"):
                outputs = {}
                async for output_name, output_data in block.run(
                    input_data, credentials=llm.TEST_CREDENTIALS
                ):
                    outputs[output_name] = output_data

            print(f"Actual calls made: {call_count}")
            print(f"Block stats: {block.execution_stats}")
            print(f"LLM call count: {block.execution_stats.llm_call_count}")

            # Should have made 2 calls: 1 for chunk summary + 1 for final summary
            assert block.execution_stats.llm_call_count >= 1
            assert block.execution_stats.input_token_count > 0
            assert block.execution_stats.output_token_count > 0

    @pytest.mark.asyncio
    async def test_ai_conversation_block_tracks_stats(self):
        """Test that AIConversationBlock correctly tracks stats."""
        import backend.blocks.llm as llm

        block = llm.AIConversationBlock()

        # Mock the llm_call method
        async def mock_llm_call(input_data, credentials):
            block.execution_stats = NodeExecutionStats(
                input_token_count=100,
                output_token_count=50,
                llm_call_count=1,
            )
            return {"response": "AI response to conversation"}

        block.llm_call = mock_llm_call  # type: ignore

        # Run the block
        input_data = llm.AIConversationBlock.Input(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        outputs = {}
        async for output_name, output_data in block.run(
            input_data, credentials=llm.TEST_CREDENTIALS
        ):
            outputs[output_name] = output_data

        # Check stats
        assert block.execution_stats.input_token_count == 100
        assert block.execution_stats.output_token_count == 50
        assert block.execution_stats.llm_call_count == 1

        # Check output
        assert outputs["response"] == "AI response to conversation"

    @pytest.mark.asyncio
    async def test_ai_list_generator_with_retries(self):
        """Test that AIListGeneratorBlock correctly tracks stats with retries."""
        import backend.blocks.llm as llm

        block = llm.AIListGeneratorBlock()

        # Counter to track calls
        call_count = 0

        async def mock_llm_call(input_data, credentials):
            nonlocal call_count
            call_count += 1

            # Update stats
            if hasattr(block, "execution_stats") and block.execution_stats:
                block.execution_stats.input_token_count += 40
                block.execution_stats.output_token_count += 20
                block.execution_stats.llm_call_count += 1
            else:
                block.execution_stats = NodeExecutionStats(
                    input_token_count=40,
                    output_token_count=20,
                    llm_call_count=1,
                )

            if call_count == 1:
                # First call returns invalid format
                return {"response": "not a valid list"}
            else:
                # Second call returns valid list
                return {"response": "['item1', 'item2', 'item3']"}

        block.llm_call = mock_llm_call  # type: ignore

        # Run the block
        input_data = llm.AIListGeneratorBlock.Input(
            focus="test items",
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            max_retries=3,
        )

        outputs = {}
        async for output_name, output_data in block.run(
            input_data, credentials=llm.TEST_CREDENTIALS
        ):
            outputs[output_name] = output_data

        # Check stats - should have 2 calls
        assert call_count == 2
        assert block.execution_stats.input_token_count == 80  # 40 * 2
        assert block.execution_stats.output_token_count == 40  # 20 * 2
        assert block.execution_stats.llm_call_count == 2

        # Check output
        assert outputs["generated_list"] == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_merge_llm_stats(self):
        """Test the merge_llm_stats method correctly merges stats from another block."""
        import backend.blocks.llm as llm

        block1 = llm.AITextGeneratorBlock()
        block2 = llm.AIStructuredResponseGeneratorBlock()

        # Set stats on block2
        block2.execution_stats = NodeExecutionStats(
            input_token_count=100,
            output_token_count=50,
            llm_call_count=2,
            llm_retry_count=1,
        )
        block2.prompt = [{"role": "user", "content": "Test"}]

        # Merge stats from block2 into block1
        block1.merge_llm_stats(block2)

        # Check that stats were merged
        assert block1.execution_stats.input_token_count == 100
        assert block1.execution_stats.output_token_count == 50
        assert block1.execution_stats.llm_call_count == 2
        assert block1.execution_stats.llm_retry_count == 1
        assert block1.prompt == [{"role": "user", "content": "Test"}]

    @pytest.mark.asyncio
    async def test_stats_initialization(self):
        """Test that blocks properly initialize stats when not present."""
        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()

        # Initially stats should be initialized with zeros
        assert hasattr(block, "execution_stats")
        assert block.execution_stats.llm_call_count == 0

        # Mock llm_call
        async def mock_llm_call(*args, **kwargs):
            return llm.LLMResponse(
                raw_response="",
                prompt=[],
                response='<json_output id="test123456">{"result": "test"}</json_output>',
                tool_calls=None,
                prompt_tokens=10,
                completion_tokens=20,
                reasoning=None,
            )

        block.llm_call = mock_llm_call  # type: ignore

        # Run the block
        input_data = llm.AIStructuredResponseGeneratorBlock.Input(
            prompt="Test",
            expected_format={"result": "desc"},
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        # Run the block
        outputs = {}
        # Mock secrets.token_hex to return consistent ID
        with patch("secrets.token_hex", return_value="test123456"):
            async for output_name, output_data in block.run(
                input_data, credentials=llm.TEST_CREDENTIALS
            ):
                outputs[output_name] = output_data

        # Block finished - now grab and assert stats
        assert block.execution_stats is not None
        assert block.execution_stats.input_token_count == 10
        assert block.execution_stats.output_token_count == 20
        assert block.execution_stats.llm_call_count == 1  # Should have exactly 1 call

        # Check output
        assert "response" in outputs
        assert outputs["response"] == {"result": "test"}


class TestAITextSummarizerValidation:
    """Test that AITextSummarizerBlock validates LLM responses are strings."""

    @pytest.mark.asyncio
    async def test_summarize_chunk_rejects_list_response(self):
        """Test that _summarize_chunk raises ValueError when LLM returns a list instead of string."""
        import backend.blocks.llm as llm

        block = llm.AITextSummarizerBlock()

        # Mock llm_call to return a list instead of a string
        async def mock_llm_call(input_data, credentials):
            # Simulate LLM returning a list when it should return a string
            return {"summary": ["bullet point 1", "bullet point 2", "bullet point 3"]}

        block.llm_call = mock_llm_call  # type: ignore

        # Create input data
        input_data = llm.AITextSummarizerBlock.Input(
            text="Some text to summarize",
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            style=llm.SummaryStyle.BULLET_POINTS,
        )

        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError) as exc_info:
            await block._summarize_chunk(
                "Some text to summarize",
                input_data,
                credentials=llm.TEST_CREDENTIALS,
            )

        error_message = str(exc_info.value)
        assert "Expected a string summary" in error_message
        assert "received list" in error_message
        assert "incorrectly formatted" in error_message

    @pytest.mark.asyncio
    async def test_combine_summaries_rejects_list_response(self):
        """Test that _combine_summaries raises ValueError when LLM returns a list instead of string."""
        import backend.blocks.llm as llm

        block = llm.AITextSummarizerBlock()

        # Mock llm_call to return a list instead of a string
        async def mock_llm_call(input_data, credentials):
            # Check if this is the final summary call
            if "final_summary" in input_data.expected_format:
                # Simulate LLM returning a list when it should return a string
                return {
                    "final_summary": [
                        "bullet point 1",
                        "bullet point 2",
                        "bullet point 3",
                    ]
                }
            else:
                return {"summary": "Valid summary"}

        block.llm_call = mock_llm_call  # type: ignore

        # Create input data
        input_data = llm.AITextSummarizerBlock.Input(
            text="Some text to summarize",
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            style=llm.SummaryStyle.BULLET_POINTS,
            max_tokens=1000,
        )

        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError) as exc_info:
            await block._combine_summaries(
                ["summary 1", "summary 2"],
                input_data,
                credentials=llm.TEST_CREDENTIALS,
            )

        error_message = str(exc_info.value)
        assert "Expected a string final summary" in error_message
        assert "received list" in error_message
        assert "incorrectly formatted" in error_message

    @pytest.mark.asyncio
    async def test_summarize_chunk_accepts_valid_string_response(self):
        """Test that _summarize_chunk accepts valid string responses."""
        import backend.blocks.llm as llm

        block = llm.AITextSummarizerBlock()

        # Mock llm_call to return a valid string
        async def mock_llm_call(input_data, credentials):
            return {"summary": "This is a valid string summary"}

        block.llm_call = mock_llm_call  # type: ignore

        # Create input data
        input_data = llm.AITextSummarizerBlock.Input(
            text="Some text to summarize",
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        # Should not raise any error
        result = await block._summarize_chunk(
            "Some text to summarize",
            input_data,
            credentials=llm.TEST_CREDENTIALS,
        )

        assert result == "This is a valid string summary"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_combine_summaries_accepts_valid_string_response(self):
        """Test that _combine_summaries accepts valid string responses."""
        import backend.blocks.llm as llm

        block = llm.AITextSummarizerBlock()

        # Mock llm_call to return a valid string
        async def mock_llm_call(input_data, credentials):
            return {"final_summary": "This is a valid final summary string"}

        block.llm_call = mock_llm_call  # type: ignore

        # Create input data
        input_data = llm.AITextSummarizerBlock.Input(
            text="Some text to summarize",
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            max_tokens=1000,
        )

        # Should not raise any error
        result = await block._combine_summaries(
            ["summary 1", "summary 2"],
            input_data,
            credentials=llm.TEST_CREDENTIALS,
        )

        assert result == "This is a valid final summary string"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_summarize_chunk_rejects_dict_response(self):
        """Test that _summarize_chunk raises ValueError when LLM returns a dict instead of string."""
        import backend.blocks.llm as llm

        block = llm.AITextSummarizerBlock()

        # Mock llm_call to return a dict instead of a string
        async def mock_llm_call(input_data, credentials):
            return {"summary": {"nested": "object", "with": "data"}}

        block.llm_call = mock_llm_call  # type: ignore

        # Create input data
        input_data = llm.AITextSummarizerBlock.Input(
            text="Some text to summarize",
            model=llm.LlmModel.GPT4O,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
        )

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            await block._summarize_chunk(
                "Some text to summarize",
                input_data,
                credentials=llm.TEST_CREDENTIALS,
            )

        error_message = str(exc_info.value)
        assert "Expected a string summary" in error_message
        assert "received dict" in error_message
