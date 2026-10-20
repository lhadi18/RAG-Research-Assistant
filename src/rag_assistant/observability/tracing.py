"""Optional Phoenix export of metadata-only OpenTelemetry spans."""
from contextlib import contextmanager
import logging
from rag_assistant.config import get_settings

logger = logging.getLogger(__name__)
_tracer = None
_attempted = False


def initialize():
    global _tracer, _attempted
    if _attempted: return _tracer is not None
    _attempted = True
    settings = get_settings()
    if not settings.phoenix_enabled: return False
    try:
        from phoenix.otel import register
        endpoint = settings.phoenix_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"): endpoint += "/v1/traces"
        provider = register(project_name="local-rag-research-assistant", endpoint=endpoint, auto_instrument=False)
        _tracer = provider.get_tracer("rag_assistant")
        logger.info("Phoenix tracing initialized at configured local endpoint")
        return True
    except Exception as exc:
        logger.warning("Phoenix tracing disabled because initialization failed: %s", type(exc).__name__)
        return False


@contextmanager
def span(name: str, attributes: dict | None = None):
    if _tracer is None: initialize()
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in (attributes or {}).items():
            if value is not None: current.set_attribute(key, value)
        yield current
