import tweepy


def handle_tweepy_exception(e: Exception) -> str:
    if isinstance(e, tweepy.BadRequest):
        return f"Bad Request (400): {e!s}"
    elif isinstance(e, tweepy.Unauthorized):
        return f"Unauthorized (401): {e!s}"
    elif isinstance(e, tweepy.Forbidden):
        return f"Forbidden (403): {e!s}"
    elif isinstance(e, tweepy.NotFound):
        return f"Not Found (404): {e!s}"
    elif isinstance(e, tweepy.TooManyRequests):
        return f"Too Many Requests (429): {e!s}"
    elif isinstance(e, tweepy.TwitterServerError):
        return f"Twitter Server Error (5xx): {e!s}"
    elif isinstance(e, tweepy.TweepyException):
        return f"Tweepy Error: {e!s}"
    else:
        return f"Unexpected error: {e!s}"
