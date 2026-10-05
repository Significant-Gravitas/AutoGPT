from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_trace_export_session() -> Session:
    """Retry transport failures when exporting OTLP span batches."""
    session = Session()
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=2,
            connect=2,
            read=2,
            status=0,
            other=0,
            allowed_methods=frozenset({"POST"}),
            backoff_factor=0.5,
            respect_retry_after_header=False,
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
