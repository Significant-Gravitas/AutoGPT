from tenacity import retry, stop_after_attempt, wait_exponential

conn_retry = retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
