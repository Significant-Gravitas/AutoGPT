from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .constant import TOO_BIG_SEQUENCE
from .utils import iana_name, is_multi_byte_encoding, unicode_range


class CharsetMatch:
    def __init__(
        self,
        payload: bytes,
        guessed_encoding: str,
        mean_mess_ratio: float,
        has_sig_or_bom: bool,
        languages: "CoherenceMatches",
        decoded_payload: Optional[str] = None,
    ):
        self._payload: bytes = payload

        self._encoding: str = guessed_encoding
        self._mean_mess_ratio: float = mean_mess_ratio
        self._languages: CoherenceMatches = languages
        self._has_sig_or_bom: bool = has_sig_or_bom
        self._unicode_ranges: Optional[List[str]] = None

        self._leaves: List[CharsetMatch] = []
        self._mean_coherence_ratio: float = 0.0

        self._output_payload: Optional[bytes] = None
        self._output_encoding: Optional[str] = None

        self._string: Optional[str] = decoded_payload

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CharsetMatch):
            raise TypeError(
                "__eq__ cannot be invoked on {} and {}.".format(
                    str(other.__class__), str(self.__class__)
                )
            )
        return self.encoding == other.encoding and self.fingerprint == other.fingerprint

    def __lt__(self, other: object) -> bool:
        """
        Implemented to make sorted available upon CharsetMatches items.
        """
        if not isinstance(other, CharsetMatch):
            raise ValueError

        chaos_difference: float = abs(self.chaos - other.chaos)
        coherence_difference: float = abs(self.coherence - other.coherence)

        # Below 1% difference --> Use Coherence
        if chaos_difference < 0.01 and coherence_difference > 0.02:
            # When having a tough decision, use the result that decoded as many multi-byte as possible.
            if chaos_difference == 0.0 and self.coherence == other.coherence:
                return self.multi_byte_usage > other.multi_byte_usage
            return self.coherence > other.coherence

        return self.chaos < other.chaos

    @property
    def multi_byte_usage(self) -> float:
        return 1.0 - len(str(self)) / len(self.raw)

    def __str__(self) -> str:
        # Lazy Str Loading
        if self._string is None:
            self._string = str(self._payload, self._encoding, "strict")
        return self._string

    def __repr__(self) -> str:
        return "<CharsetMatch '{}' bytes({})>".format(self.encoding, self.fingerprint)

    def add_submatch(self, other: "CharsetMatch") -> None:
        if not isinstance(other, CharsetMatch) or other == self:
            raise ValueError(
                "Unable to add instance <{}> as a submatch of a CharsetMatch".format(
                    other.__class__
                )
            )

        other._string = None  # Unload RAM usage; dirty trick.
        self._leaves.append(other)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def encoding_aliases(self) -> List[str]:
        """
        Encoding name are known by many name, using this could help when searching for IBM855 when it's listed as CP855.
        """
        also_known_as: List[str] = []
        for u, p in aliases.items():
            if self.encoding == u:
                also_known_as.append(p)
            elif self.encoding == p:
                also_known_as.append(u)
        return also_known_as

    @property
    def bom(self) -> bool:
        return self._has_sig_or_bom

    @property
    def byte_order_mark(self) -> bool:
        return self._has_sig_or_bom

    @property
    def languages(self) -> List[str]:
        """
        Return the complete list of possible languages found in decoded sequence.
        Usually not really useful. Returned list may be empty even if 'language' property return something != 'Unknown'.
        """
        return [e[0] for e in self._languages]

    @property
    def language(self) -> str:
        """
        Most probable language found in decoded sequence. If none were detected or inferred, the property will return
        "Unknown".
        """
        if not self._languages:
            # Trying to infer the language based on the given encoding
            # Its either English or we should not pronounce ourselves in certain cases.
            if "ascii" in self.could_be_from_charset:
                return "English"

            # doing it there to avoid circular import
            from charset_normalizer.cd import encoding_languages, mb_encoding_languages

            languages = (
                mb_encoding_languages(self.encoding)
                if is_multi_byte_encoding(self.encoding)
                else encoding_languages(self.encoding)
            )

            if len(languages) == 0 or "Latin Based" in languages:
                return "Unknown"

            return languages[0]

        return self._languages[0][0]

    @property
    def chaos(self) -> float:
        return self._mean_mess_ratio

    @property
    def coherence(self) -> float:
        if not self._languages:
            return 0.0
        return self._languages[0][1]

    @property
    def percent_chaos(self) -> float:
        return round(self.chaos * 100, ndigits=3)

    @property
    def percent_coherence(self) -> float:
        return round(self.coherence * 100, ndigits=3)

    @property
    def raw(self) -> bytes:
        """
        Original untouched bytes.
        """
        return self._payload

    @property
    def submatch(self) -> List["CharsetMatch"]:
        return self._leaves

    @property
    def has_submatch(self) -> bool:
        return len(self._leaves) > 0

    @property
    def alphabets(self) -> List[str]:
        if self._unicode_ranges is not None:
            return self._unicode_ranges
        # list detected ranges
        detected_ranges: List[Optional[str]] = [
            unicode_range(char) for char in str(self)
        ]
        # filter and sort
        self._unicode_ranges = sorted(list({r for r in detected_ranges if r}))
        return self._unicode_ranges

    @property
    def could_be_from_charset(self) -> List[str]:
        """
        The complete list of encoding that output the exact SAME str result and therefore could be the originating
        encoding.
        This list does include the encoding available in property 'encoding'.
        """
        return [self._encoding] + [m.encoding for m in self._leaves]

    def output(self, encoding: str = "utf_8") -> bytes:
        """
        Method to get re-encoded bytes payload using given target encoding. Default to UTF-8.
        Any errors will be simply ignored by the encoder NOT replaced.
        """
        if self._output_encoding is None or self._output_encoding != encoding:
            self._output_encoding = encoding
            self._output_payload = str(self).encode(encoding, "replace")

        return self._output_payload  # type: ignore

    @property
    def fingerprint(self) -> str:
        """
        Retrieve the unique SHA256 computed using the transformed (re-encoded) payload. Not the original one.
        """
        return sha256(self.output()).hexdigest()


class CharsetMatches:
    """
    Container with every CharsetMatch items ordered by default from most probable to the less one.
    Act like a list(iterable) but does not implements all related methods.
    """

    def __init__(self, results: Optional[List[CharsetMatch]] = None):
        self._results: List[CharsetMatch] = sorted(results) if results else []

    def __iter__(self) -> Iterator[CharsetMatch]:
        yield from self._results

    def __getitem__(self, item: Union[int, str]) -> CharsetMatch:
        """
        Retrieve a single item either by its position or encoding name (alias may be used here).
        Raise KeyError upon invalid index or encoding not present in results.
        """
        if isinstance(item, int):
            return self._results[item]
        if isinstance(item, str):
            item = iana_name(item, False)
            for result in self._results:
                if item in result.could_be_from_charset:
                    return result
        raise KeyError

    def __len__(self) -> int:
        return len(self._results)

    def __bool__(self) -> bool:
        return len(self._results) > 0

    def append(self, item: CharsetMatch) -> None:
        """
        Insert a single match. Will be inserted accordingly to preserve sort.
        Can be inserted as a submatch.
        """
        if not isinstance(item, CharsetMatch):
            raise ValueError(
                "Cannot append instance '{}' to CharsetMatches".format(
                    str(item.__class__)
                )
            )
        # We should disable the submatch factoring when the input file is too heavy (conserve RAM usage)
        if len(item.raw) <= TOO_BIG_SEQUENCE:
            for match in self._results:
                if match.fingerprint == item.fingerprint and match.chaos == item.chaos:
                    match.add_submatch(item)
                    return
        self._results.append(item)
        self._results = sorted(self._results)

    def best(self) -> Optional["CharsetMatch"]:
        """
        Simply return the first match. Strict equivalent to matches[0].
        """
        if not self._results:
            return None
        return self._results[0]

    def first(self) -> Optional["CharsetMatch"]:
        """
        Redundant method, call the method best(). Kept for BC reasons.
        """
        return self.best()


CoherenceMatch = Tuple[str, float]
CoherenceMatches = List[CoherenceMatch]


class CliDetectionResult:
    def __init__(
        self,
        path: str,
        encoding: Optional[str],
        encoding_aliases: List[str],
        alternative_encodings: List[str],
        language: str,
        alphabets: List[str],
        has_sig_or_bom: bool,
        chaos: float,
        coherence: float,
        unicode_path: Optional[str],
        is_preferred: bool,
    ):
        self.path: str = path
        self.unicode_path: Optional[str] = unicode_path
        self.encoding: Optional[str] = encoding
        self.encoding_aliases: List[str] = encoding_aliases
        self.alternative_encodings: List[str] = alternative_encodings
        self.language: str = language
        self.alphabets: List[str] = alphabets
        self.has_sig_or_bom: bool = has_sig_or_bom
        self.chaos: float = chaos
        self.coherence: float = coherence
        self.is_preferred: bool = is_preferred

    @property
    def __dict__(self) -> Dict[str, Any]:  # type: ignore
        return {
            "path": self.path,
            "encoding": self.encoding,
            "encoding_aliases": self.encoding_aliases,
            "alternative_encodings": self.alternative_encodings,
            "language": self.language,
            "alphabets": self.alphabets,
            "has_sig_or_bom": self.has_sig_or_bom,
            "chaos": self.chaos,
            "coherence": self.coherence,
            "unicode_path": self.unicode_path,
            "is_preferred": self.is_preferred,
        }

    def to_json(self) -> str:
        return dumps(self.__dict__, ensure_ascii=True, indent=4)
