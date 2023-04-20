## Module `send_tweet.py`

### `send_tweet() -> str`
A function that sends a tweet.

**Arguments**:
- `tweet_text` _(str)_: Text to be tweeted.

**Returns**:
- `str`: A result from sending the tweet.

**Example usage:**

```python
from send_tweet import send_tweet

result = send_tweet("Hello, World!")
print(result)
```

**Output**:
```
Tweet sent successfully!
```

or 

```
Error sending tweet: Error message.
```