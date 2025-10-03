from copy import deepcopy
from typing import Any

from tiktoken import encoding_for_model

from backend.util import json

# ---------------------------------------------------------------------------#
#  INTERNAL UTILITIES                                                         #
# ---------------------------------------------------------------------------#


def _tok_len(text: str, enc) -> int:
    """True token length of *text* in tokenizer *enc* (no wrapper cost)."""
    return len(enc.encode(str(text)))


def _msg_tokens(msg: dict, enc) -> int:
    """
    OpenAI counts ≈3 wrapper tokens per chat message, plus 1 if "name"
    is present, plus the tokenised content length.
    For tool calls, we need to count tokens in tool_calls and content fields.
    """
    WRAPPER = 3 + (1 if "name" in msg else 0)

    # Count content tokens
    content_tokens = _tok_len(msg.get("content") or "", enc)

    # Count tool call tokens for both OpenAI and Anthropic formats
    tool_call_tokens = 0

    # OpenAI format: tool_calls array at message level
    if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
        for tool_call in msg["tool_calls"]:
            # Count the tool call structure tokens
            tool_call_tokens += _tok_len(tool_call.get("id", ""), enc)
            tool_call_tokens += _tok_len(tool_call.get("type", ""), enc)
            if "function" in tool_call:
                tool_call_tokens += _tok_len(tool_call["function"].get("name", ""), enc)
                tool_call_tokens += _tok_len(
                    tool_call["function"].get("arguments", ""), enc
                )

    # Anthropic format: tool_use within content array
    content = msg.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                # Count the tool use structure tokens
                tool_call_tokens += _tok_len(item.get("id", ""), enc)
                tool_call_tokens += _tok_len(item.get("name", ""), enc)
                tool_call_tokens += _tok_len(json.dumps(item.get("input", {})), enc)
            elif isinstance(item, dict) and item.get("type") == "tool_result":
                # Count tool result tokens
                tool_call_tokens += _tok_len(item.get("tool_use_id", ""), enc)
                tool_call_tokens += _tok_len(item.get("content", ""), enc)
            elif isinstance(item, dict) and "content" in item:
                # Other content types with content field
                tool_call_tokens += _tok_len(item.get("content", ""), enc)
        # For list content, override content_tokens since we counted everything above
        content_tokens = 0

    return WRAPPER + content_tokens + tool_call_tokens


def _truncate_middle_tokens(text: str, enc, max_tok: int) -> str:
    """
    Return *text* shortened to ≈max_tok tokens by keeping the head & tail
    and inserting an ellipsis token in the middle.
    """
    ids = enc.encode(str(text))
    if len(ids) <= max_tok:
        return text  # nothing to do

    # Split the allowance between the two ends:
    head = max_tok // 2 - 1  # -1 for the ellipsis
    tail = max_tok - head - 1
    mid = enc.encode(" … ")
    return enc.decode(ids[:head] + mid + ids[-tail:])


# ---------------------------------------------------------------------------#
#  PUBLIC API                                                                 #
# ---------------------------------------------------------------------------#


def compress_prompt(
    messages: list[dict],
    target_tokens: int,
    *,
    model: str = "gpt-4o",
    reserve: int = 2_048,
    start_cap: int = 8_192,
    floor_cap: int = 128,
    lossy_ok: bool = True,
) -> list[dict]:
    """
    Shrink *messages* so that::

        token_count(prompt) + reserve  ≤  target_tokens

    Strategy
    --------
    1. **Token-aware truncation** – progressively halve a per-message cap
       (`start_cap`, `start_cap/2`, … `floor_cap`) and apply it to the
       *content* of every message except the first and last.  Tool shells
       are included: we keep the envelope but shorten huge payloads.
    2. **Middle-out deletion** – if still over the limit, delete whole
       messages working outward from the centre, **skipping** any message
       that contains ``tool_calls`` or has ``role == "tool"``.
    3. **Last-chance trim** – if still too big, truncate the *first* and
       *last* message bodies down to `floor_cap` tokens.
    4. If the prompt is *still* too large:
         • raise ``ValueError``      when ``lossy_ok == False`` (default)
         • return the partially-trimmed prompt when ``lossy_ok == True``

    Parameters
    ----------
    messages        Complete chat history (will be deep-copied).
    model           Model name; passed to tiktoken to pick the right
                    tokenizer (gpt-4o → 'o200k_base', others fallback).
    target_tokens   Hard ceiling for prompt size **excluding** the model's
                    forthcoming answer.
    reserve         How many tokens you want to leave available for that
                    answer (`max_tokens` in your subsequent completion call).
    start_cap       Initial per-message truncation ceiling (tokens).
    floor_cap       Lowest cap we'll accept before moving to deletions.
    lossy_ok        If *True* return best-effort prompt instead of raising
                    after all trim passes have been exhausted.

    Returns
    -------
    list[dict]  – A *new* messages list that abides by the rules above.
    """
    enc = encoding_for_model(model)  # best-match tokenizer
    msgs = deepcopy(messages)  # never mutate caller

    def total_tokens() -> int:
        """Current size of *msgs* in tokens."""
        return sum(_msg_tokens(m, enc) for m in msgs)

    original_token_count = total_tokens()
    if original_token_count + reserve <= target_tokens:
        return msgs

    # ---- STEP 0 : normalise content --------------------------------------
    # Convert non-string payloads to strings so token counting is coherent.
    for m in msgs[1:-1]:  # keep the first & last intact
        if not isinstance(m.get("content"), str) and m.get("content") is not None:
            # Reasonable 20k-char ceiling prevents pathological blobs
            content_str = json.dumps(m["content"], separators=(",", ":"))
            if len(content_str) > 20_000:
                content_str = _truncate_middle_tokens(content_str, enc, 20_000)
            m["content"] = content_str

    # ---- STEP 1 : token-aware truncation ---------------------------------
    cap = start_cap
    while total_tokens() + reserve > target_tokens and cap >= floor_cap:
        for m in msgs[1:-1]:  # keep first & last intact
            if _tok_len(m.get("content") or "", enc) > cap:
                m["content"] = _truncate_middle_tokens(m["content"], enc, cap)
        cap //= 2  # tighten the screw

    # ---- STEP 2 : middle-out deletion -----------------------------------
    while total_tokens() + reserve > target_tokens and len(msgs) > 2:
        centre = len(msgs) // 2
        # Build a symmetrical centre-out index walk: centre, centre+1, centre-1, ...
        order = [centre] + [
            i
            for pair in zip(range(centre + 1, len(msgs) - 1), range(centre - 1, 0, -1))
            for i in pair
        ]
        removed = False
        for i in order:
            msg = msgs[i]
            if "tool_calls" in msg or msg.get("role") == "tool":
                continue  # protect tool shells
            del msgs[i]
            removed = True
            break
        if not removed:  # nothing more we can drop
            break

    # ---- STEP 3 : final safety-net trim on first & last ------------------
    cap = start_cap
    while total_tokens() + reserve > target_tokens and cap >= floor_cap:
        for idx in (0, -1):  # first and last
            text = msgs[idx].get("content") or ""
            if _tok_len(text, enc) > cap:
                msgs[idx]["content"] = _truncate_middle_tokens(text, enc, cap)
        cap //= 2  # tighten the screw

    # ---- STEP 4 : success or fail-gracefully -----------------------------
    if total_tokens() + reserve > target_tokens and not lossy_ok:
        raise ValueError(
            "compress_prompt: prompt still exceeds budget "
            f"({total_tokens() + reserve} > {target_tokens})."
        )

    return msgs


def estimate_token_count(
    messages: list[dict],
    *,
    model: str = "gpt-4o",
) -> int:
    """
    Return the true token count of *messages* when encoded for *model*.

    Parameters
    ----------
    messages    Complete chat history.
    model       Model name; passed to tiktoken to pick the right
                tokenizer (gpt-4o → 'o200k_base', others fallback).

    Returns
    -------
    int  – Token count.
    """
    enc = encoding_for_model(model)  # best-match tokenizer
    return sum(_msg_tokens(m, enc) for m in messages)


def estimate_token_count_str(
    text: Any,
    *,
    model: str = "gpt-4o",
) -> int:
    """
    Return the true token count of *text* when encoded for *model*.

    Parameters
    ----------
    text    Input text.
    model   Model name; passed to tiktoken to pick the right
            tokenizer (gpt-4o → 'o200k_base', others fallback).

    Returns
    -------
    int  – Token count.
    """
    enc = encoding_for_model(model)  # best-match tokenizer
    text = json.dumps(text) if not isinstance(text, str) else text
    return _tok_len(text, enc)
