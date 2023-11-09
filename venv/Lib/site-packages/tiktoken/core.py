from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union

import regex

from tiktoken import _tiktoken


class Encoding:
    def __init__(
        self,
        name: str,
        *,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
        explicit_n_vocab: Optional[int] = None,
    ):
        """Creates an Encoding object.

        See openai_public.py for examples of how to construct an Encoding object.

        Args:
            name: The name of the encoding. It should be clear from the name of the encoding
                what behaviour to expect, in particular, encodings with different special tokens
                should have different names.
            pat_str: A regex pattern string that is used to split the input text.
            mergeable_ranks: A dictionary mapping mergeable token bytes to their ranks. The ranks
                must correspond to merge priority.
            special_tokens: A dictionary mapping special token strings to their token values.
            explicit_n_vocab: The number of tokens in the vocabulary. If provided, it is checked
                that the number of mergeable tokens and special tokens is equal to this number.
        """
        self.name = name

        self._pat_str = pat_str
        self._mergeable_ranks = mergeable_ranks
        self._special_tokens = special_tokens

        self.max_token_value = max(
            max(mergeable_ranks.values()), max(special_tokens.values(), default=0)
        )
        if explicit_n_vocab:
            assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
            assert self.max_token_value == explicit_n_vocab - 1

        self._core_bpe = _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)

    def __repr__(self) -> str:
        return f"<Encoding {self.name!r}>"

    # ====================
    # Encoding
    # ====================

    def encode_ordinary(self, text: str) -> list[int]:
        """Encodes a string into tokens, ignoring special tokens.

        This is equivalent to `encode(text, disallowed_special=())` (but slightly faster).

        ```
        >>> enc.encode_ordinary("hello world")
        [31373, 995]
        """
        try:
            return self._core_bpe.encode_ordinary(text)
        except UnicodeEncodeError:
            # See comment in encode
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._core_bpe.encode_ordinary(text)

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        try:
            return self._core_bpe.encode(text, allowed_special)
        except UnicodeEncodeError:
            # BPE operates on bytes, but the regex operates on unicode. If we pass a str that is
            # invalid UTF-8 to Rust, it will rightfully complain. Here we do a quick and dirty
            # fixup for any surrogate pairs that may have sneaked their way into the text.
            # Technically, this introduces a place where encode + decode doesn't roundtrip a Python
            # string, but given that this is input we want to support, maybe that's okay.
            # Also we use errors="replace" to handle weird things like lone surrogates.
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
            return self._core_bpe.encode(text, allowed_special)

    def encode_ordinary_batch(self, text: list[str], *, num_threads: int = 8) -> list[list[int]]:
        """Encodes a list of strings into tokens, in parallel, ignoring special tokens.

        This is equivalent to `encode_batch(text, disallowed_special=())` (but slightly faster).

        ```
        >>> enc.encode_ordinary_batch(["hello world", "goodbye world"])
        [[31373, 995], [11274, 16390, 995]]
        ```
        """
        encoder = functools.partial(self.encode_ordinary)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def encode_batch(
        self,
        text: list[str],
        *,
        num_threads: int = 8,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> list[list[int]]:
        """Encodes a list of strings into tokens, in parallel.

        See `encode` for more details on `allowed_special` and `disallowed_special`.

        ```
        >>> enc.encode_batch(["hello world", "goodbye world"])
        [[31373, 995], [11274, 16390, 995]]
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        if not isinstance(disallowed_special, frozenset):
            disallowed_special = frozenset(disallowed_special)

        encoder = functools.partial(
            self.encode, allowed_special=allowed_special, disallowed_special=disallowed_special
        )
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def encode_with_unstable(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),  # noqa: B006
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> tuple[list[int], list[list[int]]]:
        """Encodes a string into stable tokens and possible completion sequences.

        Note that the stable tokens will only represent a substring of `text`.

        See `encode` for more details on `allowed_special` and `disallowed_special`.

        This API should itself be considered unstable.

        ```
        >>> enc.encode_with_unstable("hello fanta")
        ([31373], [(277, 4910), (5113, 265), ..., (8842,)])

        >>> text = "..."
        >>> stable_tokens, completions = enc.encode_with_unstable(text)
        >>> assert text.encode().startswith(enc.decode_bytes(stable_tokens))
        >>> assert all(enc.decode_bytes(stable_tokens + seq).startswith(text.encode()) for seq in completions)
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        return self._core_bpe.encode_with_unstable(text, allowed_special)

    def encode_single_token(self, text_or_bytes: Union[str, bytes]) -> int:
        """Encodes text corresponding to a single token to its token value.

        NOTE: this will encode all special tokens.

        Raises `KeyError` if the token is not in the vocabulary.

        ```
        >>> enc.encode_single_token("hello")
        31373
        ```
        """
        if isinstance(text_or_bytes, str):
            text_or_bytes = text_or_bytes.encode("utf-8")
        return self._core_bpe.encode_single_token(text_or_bytes)

    # ====================
    # Decoding
    # ====================

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        ```
        >>> enc.decode_bytes([31373, 995])
        b'hello world'
        ```
        """
        return self._core_bpe.decode_bytes(tokens)

    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        """Decodes a list of tokens into a string.

        WARNING: the default behaviour of this function is lossy, since decoded bytes are not
        guaranteed to be valid UTF-8. You can control this behaviour using the `errors` parameter,
        for instance, setting `errors=strict`.

        ```
        >>> enc.decode([31373, 995])
        'hello world'
        ```
        """
        return self._core_bpe.decode_bytes(tokens).decode("utf-8", errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Decodes a token into bytes.

        NOTE: this will decode all special tokens.

        Raises `KeyError` if the token is not in the vocabulary.

        ```
        >>> enc.decode_single_token_bytes(31373)
        b'hello'
        ```
        """
        return self._core_bpe.decode_single_token_bytes(token)

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising tokenisation.
        >>> enc.decode_tokens_bytes([31373, 995])
        [b'hello', b' world']
        """
        return [self.decode_single_token_bytes(token) for token in tokens]

    # ====================
    # Miscellaneous
    # ====================

    def token_byte_values(self) -> list[bytes]:
        """Returns the list of all token byte values."""
        return self._core_bpe.token_byte_values()

    @property
    def eot_token(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    @functools.cached_property
    def special_tokens_set(self) -> set[str]:
        return set(self._special_tokens.keys())

    @property
    def n_vocab(self) -> int:
        """For backwards compatibility. Prefer to use `enc.max_token_value + 1`."""
        return self.max_token_value + 1

    # ====================
    # Private
    # ====================

    def _encode_single_piece(self, text_or_bytes: Union[str, bytes]) -> list[int]:
        """Encodes text corresponding to bytes without a regex split.

        NOTE: this will not encode any special tokens.

        ```
        >>> enc.encode_single_piece("helloqqqq")
        [31373, 38227, 38227]
        ```
        """
        if isinstance(text_or_bytes, str):
            text_or_bytes = text_or_bytes.encode("utf-8")
        return self._core_bpe.encode_single_piece(text_or_bytes)

    def _encode_only_native_bpe(self, text: str) -> list[int]:
        """Encodes a string into tokens, but do regex splitting in Python."""
        _unused_pat = regex.compile(self._pat_str)
        ret = []
        for piece in regex.findall(_unused_pat, text):
            ret.extend(self._core_bpe.encode_single_piece(piece))
        return ret

    def _encode_bytes(self, text: bytes) -> list[int]:
        return self._core_bpe._encode_bytes(text)


@functools.lru_cache(maxsize=128)
def _special_token_regex(tokens: frozenset[str]) -> "regex.Pattern[str]":
    inner = "|".join(regex.escape(token) for token in tokens)
    return regex.compile(f"({inner})")


def raise_disallowed_special_token(token: str) -> NoReturn:
    raise ValueError(
        f"Encountered text corresponding to disallowed special token {token!r}.\n"
        "If you want this text to be encoded as a special token, "
        f"pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\n"
        f"If you want this text to be encoded as normal text, disable the check for this token "
        f"by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\n"
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
    )
