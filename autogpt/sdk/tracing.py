import os
from functools import wraps

from dotenv import load_dotenv

from autogpt.sdk.forge_log import ForgeLogger

load_dotenv()

ENABLE_TRACING = os.environ.get("ENABLE_TRACING", "false").lower() == "true"

LOG = ForgeLogger(__name__)


def setup_tracing(app):
    LOG.info(f"Tracing status: {ENABLE_TRACING}")

    if ENABLE_TRACING:
        from opentelemetry import trace
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(attributes={SERVICE_NAME: "Auto-GPT-Forge"})

        # Configure the tracer provider to export traces to Jaeger
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger",
            agent_port=6831,
        )

        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Instrument FastAPI app
        # FastAPIInstrumentor.instrument_app(app)
        LOG.info("Tracing Setup")


if ENABLE_TRACING:
    from functools import wraps

    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan
    from pydantic import BaseModel

    from autogpt.sdk.schema import Task

    tasks_context_db = {}

    class TaskIDTraceContext:
        """Custom context manager to override default tracing behavior."""

        def __init__(self, task_id: str, span_name: str):
            self.task_id = task_id
            self.span_name = span_name
            self.span = None

        def __enter__(self):
            # Get the default tracer
            tracer = trace.get_tracer(__name__)

            # Check if the task_id has been traced before
            if self.task_id in tasks_context_db:
                # Get the span context from the previous task
                span_context = tasks_context_db[self.task_id].get("span_context")
                LOG.info(
                    f"Task ID: {self.task_id} Span Context trace_id: {span_context.trace_id} span_id: {span_context.span_id}"
                )
                assert span_context, "No Span context for existing task_id"
                # Create a new span linked to the previous span context
                ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
                self.span = tracer.start_span(self.span_name, context=ctx)
            else:
                # If it's a new task_id, start a new span
                self.span = tracer.start_span(self.span_name)
                # Set this span in context and store it for future use
                tasks_context_db[self.task_id] = {
                    "span_context": self.span.get_span_context()
                }

            return self.span

        def __exit__(self, type, value, traceback):
            if self.span:
                self.span.end()
                self.span = None

    def tracing(operation_name: str, is_create_task: bool = False):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                function_ran = False
                task_id = "none"
                if is_create_task:
                    result = await func(*args, **kwargs)
                    if isinstance(result, Task):
                        task_id = result.task_id
                    function_ran = True
                else:
                    task_id = kwargs.get("task_id", "none")
                step_id = kwargs.get("step_id", "none")
                LOG.info(f"Starting Trace for task_id: {task_id}")

                with TaskIDTraceContext(task_id, operation_name) as span:
                    span.set_attribute("task_id", task_id)
                    span.set_attribute("step_id", step_id)
                    # Add request event with all kwargs
                    kwargs_copy = {k: v for k, v in kwargs.items() if k != "request"}
                    for key, value in kwargs_copy.items():
                        if isinstance(value, BaseModel):
                            kwargs_copy[key] = value.json()

                    span.add_event(name="request", attributes=kwargs_copy)
                    if not function_ran:
                        result = await func(*args, **kwargs)
                    # Convert result to json before adding response event
                    if isinstance(result, BaseModel):
                        result_json = result.json()
                        span.add_event("response", {"response": result_json})
                    return result

            return wrapper

        return decorator

else:

    def tracing(operation_name: str, is_create_task: bool = False):
        """
        Stub function that does nothing so we can have a global enable tracing switch
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator
