"""
observability/otel.py

Single configuration point for OpenTelemetry.
Sets up a TracerProvider with an OTLP gRPC exporter pointing at Jaeger.

All nodes call get_tracer() to open spans. If OTLP_ENDPOINT is absent
from the environment, a no-op tracer is returned — the graph runs
identically with tracing silently disabled.

Exports:
    setup_otel()     — called once from clients.py at startup
    get_tracer()     — returns the configured tracer
    traced_node()    — decorator that wraps a node function in an OTel span
"""

import os
import logging
import functools
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

_tracer: trace.Tracer | None = None


def setup_otel(service_name: str = "industrial-maintenance-ai-v7") -> None:
    """
    Initialise the global TracerProvider.
    Called once at application startup from nodes/clients.py.

    If OTLP_ENDPOINT is not set, OTel is left unconfigured and
    get_tracer() returns a no-op tracer.
    """
    global _tracer

    endpoint = os.getenv("OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OTLP_ENDPOINT not set — OTel tracing disabled")
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

    logger.info("OTel tracing initialised → %s", endpoint)


def get_tracer() -> trace.Tracer:
    """
    Return the configured tracer.
    If setup_otel() was not called or OTLP_ENDPOINT was absent,
    returns a no-op tracer that produces zero overhead.
    """
    if _tracer is not None:
        return _tracer
    return trace.get_tracer("noop")


def traced_node(span_name: str):
    """
    Decorator that wraps a LangGraph node function in an OTel span.

    Usage:
        @traced_node("node.memory_retrieval")
        def memory_retrieval_node(state, span) -> dict:
            ...
            span.set_attribute("episodes_retrieved", len(memories))
            return {...}

    What runs before the original function:
        - get_tracer() fetches the configured tracer
        - start_as_current_span() opens the span and starts the timer in Jaeger

    What runs after the original function:
        - span closes automatically, stopping the timer
        - span is sent to Jaeger via the BatchSpanProcessor
        - if the node raised an exception:
            - the exception is recorded on the span
            - the span status is set to ERROR
            - the exception is re-raised so LangGraph still sees it

    The node function receives (state, span) as arguments:
        - state  — the AgentState passed by LangGraph (unchanged)
        - span   — the open OTel span, used to set node-specific attributes
                   nodes that have nothing to record can simply ignore it

    @functools.wraps(fn) ensures the original function name and docstring
    are preserved on wrapper — LangGraph uses function names internally
    during graph construction.
    """
    def decorator(fn):                          # receives the original node function

        @functools.wraps(fn)                    # preserves fn.__name__ on wrapper
        def wrapper(state):                     # LangGraph calls this
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = fn(state, span)    # calls the original node function
                    return result
                except Exception as e:
                    span.set_status(           # mark span as ERROR in Jaeger
                        trace.StatusCode.ERROR,
                        str(e),
                    )
                    span.record_exception(e)   # attach full traceback to span
                    raise                      # re-raise so LangGraph sees it

        return wrapper                          # decorator returns wrapper

    return decorator                            # traced_node returns decorator
