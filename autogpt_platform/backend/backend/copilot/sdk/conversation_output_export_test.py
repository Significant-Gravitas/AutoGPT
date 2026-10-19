import asyncio
import json

import pytest
from langsmith import Client, tracing_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from . import conversation_output_test as fixtures
from .conversation_output_test import assistant, consume, result

blocked = fixtures.blocked
client = fixtures.client
script = fixtures.script


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_output_reaches_otel_export(client, script, cancelled):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing_client = Client(tracing_mode="otel", otel_tracer_provider=provider)
    script.extend(
        [
            assistant("exported reply"),
            asyncio.CancelledError() if cancelled else result(),
        ]
    )
    try:
        with tracing_context(enabled=True, client=tracing_client):
            if cancelled:
                with pytest.raises(asyncio.CancelledError):
                    await consume(client)
            else:
                await consume(client)
        await asyncio.to_thread(tracing_client.flush, timeout=5)
        spans = [
            span
            for span in exporter.get_finished_spans()
            if span.name == "claude.conversation"
        ]
        assert len(spans) == 1
        assert json.loads(spans[0].attributes["gen_ai.completion"]) == {
            "output": "exported reply",
            "stop_reason": "cancelled" if cancelled else "end_turn",
        }
    finally:
        tracing_client.close()
        provider.shutdown()
