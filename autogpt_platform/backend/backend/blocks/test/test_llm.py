from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import openai
import pytest

import backend.blocks.llm as llm
from backend.data.model import NodeExecutionStats

# TEST_CREDENTIALS_INPUT is a plain dict that satisfies AICredentials at runtime
# but not at the type level. Cast once here to avoid per-test suppressors.
_TEST_AI_CREDENTIALS = cast(llm.AICredentials, llm.TEST_CREDENTIALS_INPUT)


class TestLLMStatsTracking:
    """Test that LLM blocks correctly track token usage statistics."""

    @pytest.mark.asyncio
    async def test_llm_call_returns_token_counts(self):
        """Test that llm_call returns proper token counts in LLMResponse."""
        import backend.blocks.llm as llm

        # Mock the OpenAI Responses API response
        mock_response = MagicMock()
        mock_response.output_text = "Test response"
        mock_response.output = []
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)

        # Test with mocked OpenAI response
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            response = await llm.llm_call(
                credentials=llm.TEST_CREDENTIALS,
                llm_model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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
    async def test_retry_cost_uses_last_attempt_only(self):
        """provider_cost is only merged from the final successful attempt.

        Intermediate retry costs are intentionally dropped to avoid
        double-counting: the cost of failed attempts is captured in
        last_attempt_cost only when the loop eventually succeeds.
        """
        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()
        call_count = 0

        async def mock_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: fails validation, returns cost $0.01
                return llm.LLMResponse(
                    raw_response="",
                    prompt=[],
                    response='<json_output id="test123456">{"wrong": "key"}</json_output>',
                    tool_calls=None,
                    prompt_tokens=10,
                    completion_tokens=5,
                    reasoning=None,
                    provider_cost=0.01,
                )
            # Second attempt: succeeds, returns cost $0.02
            return llm.LLMResponse(
                raw_response="",
                prompt=[],
                response='<json_output id="test123456">{"key1": "value1", "key2": "value2"}</json_output>',
                tool_calls=None,
                prompt_tokens=20,
                completion_tokens=10,
                reasoning=None,
                provider_cost=0.02,
            )

        block.llm_call = mock_llm_call  # type: ignore

        input_data = llm.AIStructuredResponseGeneratorBlock.Input(
            prompt="Test prompt",
            expected_format={"key1": "desc1", "key2": "desc2"},
            model=llm.DEFAULT_LLM_MODEL,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            retry=2,
        )

        with patch("secrets.token_hex", return_value="test123456"):
            async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                pass

        # Only the final successful attempt's cost is merged
        assert block.execution_stats.provider_cost == pytest.approx(0.02)
        # Tokens from both attempts accumulate
        assert block.execution_stats.input_token_count == 30
        assert block.execution_stats.output_token_count == 15

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
            model=llm.DEFAULT_LLM_MODEL,
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
                mock_response.output_text = '<json_output id="test123456">{"summary": "Test chunk summary"}</json_output>'
            else:
                mock_response.output_text = '<json_output id="test123456">{"final_summary": "Test final summary"}</json_output>'
            mock_response.output = []
            mock_response.usage = MagicMock(input_tokens=50, output_tokens=30)
            return mock_response

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.responses.create = mock_create

            # Test with very short text (should only need 1 chunk + 1 final summary)
            input_data = llm.AITextSummarizerBlock.Input(
                text="This is a short text.",
                model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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
    async def test_ai_list_generator_basic_functionality(self):
        """Test that AIListGeneratorBlock correctly works with structured responses."""
        import backend.blocks.llm as llm

        block = llm.AIListGeneratorBlock()

        # Mock the llm_call to return a structured response
        async def mock_llm_call(input_data, credentials):
            # Update stats to simulate LLM call
            block.execution_stats = NodeExecutionStats(
                input_token_count=50,
                output_token_count=30,
                llm_call_count=1,
            )
            # Return a structured response with the expected format
            return {"list": ["item1", "item2", "item3"]}

        block.llm_call = mock_llm_call  # type: ignore

        # Run the block
        input_data = llm.AIListGeneratorBlock.Input(
            focus="test items",
            model=llm.DEFAULT_LLM_MODEL,
            credentials=llm.TEST_CREDENTIALS_INPUT,  # type: ignore
            max_retries=3,
        )

        outputs = {}
        async for output_name, output_data in block.run(
            input_data, credentials=llm.TEST_CREDENTIALS
        ):
            outputs[output_name] = output_data

        # Check stats
        assert block.execution_stats.input_token_count == 50
        assert block.execution_stats.output_token_count == 30
        assert block.execution_stats.llm_call_count == 1

        # Check output
        assert outputs["generated_list"] == ["item1", "item2", "item3"]
        # Check that individual items were yielded
        # Note: outputs dict will only contain the last value for each key
        # So we need to check that the list_item output exists
        assert "list_item" in outputs
        # The list_item output should be the last item in the list
        assert outputs["list_item"] == "item3"
        assert "prompt" in outputs

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
            model=llm.DEFAULT_LLM_MODEL,
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


class TestAIConversationBlockValidation:
    """Test that AIConversationBlock validates inputs before calling the LLM."""

    @pytest.mark.asyncio
    async def test_empty_messages_and_empty_prompt_raises_error(self):
        """Empty messages with no prompt should raise ValueError, not a cryptic API error."""
        block = llm.AIConversationBlock()

        input_data = llm.AIConversationBlock.Input(
            messages=[],
            prompt="",
            model=llm.DEFAULT_LLM_MODEL,
            credentials=_TEST_AI_CREDENTIALS,
        )

        with pytest.raises(ValueError, match="no messages and no prompt"):
            async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                pass

    @pytest.mark.asyncio
    async def test_empty_messages_with_prompt_succeeds(self):
        """Empty messages but a non-empty prompt should proceed without error."""
        block = llm.AIConversationBlock()

        async def mock_llm_call(input_data, credentials):
            return {"response": "OK"}

        with patch.object(block, "llm_call", new=AsyncMock(side_effect=mock_llm_call)):
            input_data = llm.AIConversationBlock.Input(
                messages=[],
                prompt="Hello, how are you?",
                model=llm.DEFAULT_LLM_MODEL,
                credentials=_TEST_AI_CREDENTIALS,
            )

            outputs = {}
            async for name, data in block.run(
                input_data, credentials=llm.TEST_CREDENTIALS
            ):
                outputs[name] = data

        assert outputs["response"] == "OK"

    @pytest.mark.asyncio
    async def test_nonempty_messages_with_empty_prompt_succeeds(self):
        """Non-empty messages with no prompt should proceed without error."""
        block = llm.AIConversationBlock()

        async def mock_llm_call(input_data, credentials):
            return {"response": "response from conversation"}

        with patch.object(block, "llm_call", new=AsyncMock(side_effect=mock_llm_call)):
            input_data = llm.AIConversationBlock.Input(
                messages=[{"role": "user", "content": "Hello"}],
                prompt="",
                model=llm.DEFAULT_LLM_MODEL,
                credentials=_TEST_AI_CREDENTIALS,
            )

            outputs = {}
            async for name, data in block.run(
                input_data, credentials=llm.TEST_CREDENTIALS
            ):
                outputs[name] = data

        assert outputs["response"] == "response from conversation"

    @pytest.mark.asyncio
    async def test_messages_with_empty_content_raises_error(self):
        """Messages with empty content strings should be treated as no messages."""
        block = llm.AIConversationBlock()

        input_data = llm.AIConversationBlock.Input(
            messages=[{"role": "user", "content": ""}],
            prompt="",
            model=llm.DEFAULT_LLM_MODEL,
            credentials=_TEST_AI_CREDENTIALS,
        )

        with pytest.raises(ValueError, match="no messages and no prompt"):
            async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                pass

    @pytest.mark.asyncio
    async def test_messages_with_whitespace_content_raises_error(self):
        """Messages with whitespace-only content should be treated as no messages."""
        block = llm.AIConversationBlock()

        input_data = llm.AIConversationBlock.Input(
            messages=[{"role": "user", "content": "   "}],
            prompt="",
            model=llm.DEFAULT_LLM_MODEL,
            credentials=_TEST_AI_CREDENTIALS,
        )

        with pytest.raises(ValueError, match="no messages and no prompt"):
            async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                pass

    @pytest.mark.asyncio
    async def test_messages_with_none_entry_raises_error(self):
        """Messages list containing None should be treated as no messages."""
        block = llm.AIConversationBlock()

        input_data = llm.AIConversationBlock.Input(
            messages=[None],
            prompt="",
            model=llm.DEFAULT_LLM_MODEL,
            credentials=_TEST_AI_CREDENTIALS,
        )

        with pytest.raises(ValueError, match="no messages and no prompt"):
            async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                pass

    @pytest.mark.asyncio
    async def test_messages_with_empty_dict_raises_error(self):
        """Messages list containing empty dict should be treated as no messages."""
        block = llm.AIConversationBlock()

        input_data = llm.AIConversationBlock.Input(
            messages=[{}],
            prompt="",
            model=llm.DEFAULT_LLM_MODEL,
            credentials=_TEST_AI_CREDENTIALS,
        )

        with pytest.raises(ValueError, match="no messages and no prompt"):
            async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                pass

    @pytest.mark.asyncio
    async def test_messages_with_none_content_raises_error(self):
        """Messages with content=None should not crash with AttributeError."""
        block = llm.AIConversationBlock()

        input_data = llm.AIConversationBlock.Input(
            messages=[{"role": "user", "content": None}],
            prompt="",
            model=llm.DEFAULT_LLM_MODEL,
            credentials=_TEST_AI_CREDENTIALS,
        )

        with pytest.raises(ValueError, match="no messages and no prompt"):
            async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                pass


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
            model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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
            model=llm.DEFAULT_LLM_MODEL,
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


def _make_anthropic_status_error(status_code: int) -> anthropic.APIStatusError:
    """Create an anthropic.APIStatusError with the given status code."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return anthropic.APIStatusError(
        f"Error code: {status_code}", response=response, body=None
    )


def _make_openai_status_error(status_code: int) -> openai.APIStatusError:
    """Create an openai.APIStatusError with the given status code."""
    response = httpx.Response(
        status_code, request=httpx.Request("POST", "https://api.openai.com/v1/chat")
    )
    return openai.APIStatusError(
        f"Error code: {status_code}", response=response, body=None
    )


class TestUserErrorStatusCodeHandling:
    """Test that user-caused LLM API errors (401/403/429) break the retry loop
    and are logged as warnings, while server errors (500) trigger retries."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [401, 403, 429])
    async def test_anthropic_user_error_breaks_retry_loop(self, status_code: int):
        """401/403/429 Anthropic errors should break immediately, not retry."""
        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()
        call_count = 0

        async def mock_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _make_anthropic_status_error(status_code)

        with patch.object(block, "llm_call", new=AsyncMock(side_effect=mock_llm_call)):
            input_data = llm.AIStructuredResponseGeneratorBlock.Input(
                prompt="Test",
                expected_format={"key": "desc"},
                model=llm.DEFAULT_LLM_MODEL,
                credentials=_TEST_AI_CREDENTIALS,
                retry=3,
            )

            with pytest.raises(RuntimeError):
                async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                    pass

        assert (
            call_count == 1
        ), f"Expected exactly 1 call for status {status_code}, got {call_count}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [401, 403, 429])
    async def test_openai_user_error_breaks_retry_loop(self, status_code: int):
        """401/403/429 OpenAI errors should break immediately, not retry."""
        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()
        call_count = 0

        async def mock_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _make_openai_status_error(status_code)

        with patch.object(block, "llm_call", new=AsyncMock(side_effect=mock_llm_call)):
            input_data = llm.AIStructuredResponseGeneratorBlock.Input(
                prompt="Test",
                expected_format={"key": "desc"},
                model=llm.DEFAULT_LLM_MODEL,
                credentials=_TEST_AI_CREDENTIALS,
                retry=3,
            )

            with pytest.raises(RuntimeError):
                async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                    pass

        assert (
            call_count == 1
        ), f"Expected exactly 1 call for status {status_code}, got {call_count}"

    @pytest.mark.asyncio
    async def test_server_error_retries(self):
        """500 errors should be retried (not break immediately)."""
        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()
        call_count = 0

        async def mock_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _make_anthropic_status_error(500)

        with patch.object(block, "llm_call", new=AsyncMock(side_effect=mock_llm_call)):
            input_data = llm.AIStructuredResponseGeneratorBlock.Input(
                prompt="Test",
                expected_format={"key": "desc"},
                model=llm.DEFAULT_LLM_MODEL,
                credentials=_TEST_AI_CREDENTIALS,
                retry=3,
            )

            with pytest.raises(RuntimeError):
                async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                    pass

        assert (
            call_count > 1
        ), f"Expected multiple retry attempts for 500, got {call_count}"

    @pytest.mark.asyncio
    async def test_user_error_logs_warning_not_exception(self):
        """User-caused errors should log with logger.warning, not logger.exception."""
        import backend.blocks.llm as llm

        block = llm.AIStructuredResponseGeneratorBlock()

        async def mock_llm_call(*args, **kwargs):
            raise _make_anthropic_status_error(401)

        with patch.object(block, "llm_call", new=AsyncMock(side_effect=mock_llm_call)):
            input_data = llm.AIStructuredResponseGeneratorBlock.Input(
                prompt="Test",
                expected_format={"key": "desc"},
                model=llm.DEFAULT_LLM_MODEL,
                credentials=_TEST_AI_CREDENTIALS,
            )

            with (
                patch.object(llm.logger, "warning") as mock_warning,
                patch.object(llm.logger, "exception") as mock_exception,
                pytest.raises(RuntimeError),
            ):
                async for _ in block.run(input_data, credentials=llm.TEST_CREDENTIALS):
                    pass

        mock_warning.assert_called_once()
        mock_exception.assert_not_called()


class TestLlmModelMissing:
    """Test that LlmModel handles provider-prefixed model names."""

    def test_provider_prefixed_model_resolves(self):
        """Provider-prefixed model string should resolve to the correct enum member."""
        assert (
            llm.LlmModel("anthropic/claude-sonnet-4-6")
            == llm.LlmModel.CLAUDE_4_6_SONNET
        )

    def test_bare_model_still_works(self):
        """Bare (non-prefixed) model string should still resolve correctly."""
        assert llm.LlmModel("claude-sonnet-4-6") == llm.LlmModel.CLAUDE_4_6_SONNET

    def test_invalid_prefixed_model_raises(self):
        """Unknown provider-prefixed model string should raise ValueError."""
        with pytest.raises(ValueError):
            llm.LlmModel("invalid/nonexistent-model")

    def test_slash_containing_value_direct_lookup(self):
        """Enum values with '/' (e.g., OpenRouter models) should resolve via direct lookup, not _missing_."""
        assert llm.LlmModel("google/gemini-2.5-pro") == llm.LlmModel.GEMINI_2_5_PRO

    def test_double_prefixed_slash_model(self):
        """Double-prefixed value should still resolve by stripping first prefix."""
        assert (
            llm.LlmModel("extra/google/gemini-2.5-pro") == llm.LlmModel.GEMINI_2_5_PRO
        )


class TestExtractOpenRouterCost:
    """Tests for extract_openrouter_cost — the x-total-cost header parser."""

    def _mk_response(self, headers: dict | None):
        response = MagicMock()
        if headers is None:
            response._response = None
        else:
            raw = MagicMock()
            raw.headers = headers
            response._response = raw
        return response

    def test_extracts_numeric_cost(self):
        response = self._mk_response({"x-total-cost": "0.0042"})
        assert llm.extract_openrouter_cost(response) == 0.0042

    def test_returns_none_when_header_missing(self):
        response = self._mk_response({})
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_when_header_empty_string(self):
        response = self._mk_response({"x-total-cost": ""})
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_when_header_non_numeric(self):
        response = self._mk_response({"x-total-cost": "not-a-number"})
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_when_no_response_attr(self):
        response = MagicMock(spec=[])  # no _response attr
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_when_raw_is_none(self):
        response = self._mk_response(None)
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_when_raw_has_no_headers(self):
        response = MagicMock()
        response._response = MagicMock(spec=[])  # no headers attr
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_zero_for_zero_cost(self):
        """Zero-cost is a valid value (free tier) and must not become None."""
        response = self._mk_response({"x-total-cost": "0"})
        assert llm.extract_openrouter_cost(response) == 0.0

    def test_returns_none_for_inf(self):
        response = self._mk_response({"x-total-cost": "inf"})
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_for_negative_inf(self):
        response = self._mk_response({"x-total-cost": "-inf"})
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_for_nan(self):
        response = self._mk_response({"x-total-cost": "nan"})
        assert llm.extract_openrouter_cost(response) is None

    def test_returns_none_for_negative_cost(self):
        response = self._mk_response({"x-total-cost": "-0.005"})
        assert llm.extract_openrouter_cost(response) is None
