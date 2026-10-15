import tweepy


def handle_tweepy_exception(e: Exception) -> str:
    if isinstance(e, tweepy.BadRequest):
        return f"Bad Request (400): {str(e)}"
    elif isinstance(e, tweepy.Unauthorized):
        return f"Unauthorized (401): {str(e)}"
    elif isinstance(e, tweepy.Forbidden):
        return f"Forbidden (403): {str(e)}"
    elif isinstance(e, tweepy.NotFound):
        return f"Not Found (404): {str(e)}"
    elif isinstance(e, tweepy.TooManyRequests):
        return f"Too Many Requests (429): {str(e)}"
    elif isinstance(e, tweepy.TwitterServerError):
        return f"Twitter Server Error (5xx): {str(e)}"
    elif isinstance(e, tweepy.TweepyException):
        return f"Tweepy Error: {str(e)}"
    else:
        return f"Unexpected error: {str(e)}"
