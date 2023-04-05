import codecs
import re
from string import ascii_letters, ascii_lowercase, digits
from typing import Optional, cast

BASCII_LOWERCASE = ascii_lowercase.encode("ascii")
BPCT_ALLOWED = {f"%{i:02X}".encode("ascii") for i in range(256)}
GEN_DELIMS = ":/?#[]@"
SUB_DELIMS_WITHOUT_QS = "!$'()*,"
SUB_DELIMS = SUB_DELIMS_WITHOUT_QS + "+&=;"
RESERVED = GEN_DELIMS + SUB_DELIMS
UNRESERVED = ascii_letters + digits + "-._~"
ALLOWED = UNRESERVED + SUB_DELIMS_WITHOUT_QS


_IS_HEX = re.compile(b"[A-Z0-9][A-Z0-9]")
_IS_HEX_STR = re.compile("[A-Fa-f0-9][A-Fa-f0-9]")

utf8_decoder = codecs.getincrementaldecoder("utf-8")


class _Quoter:
    def __init__(
        self,
        *,
        safe: str = "",
        protected: str = "",
        qs: bool = False,
        requote: bool = True,
    ) -> None:
        self._safe = safe
        self._protected = protected
        self._qs = qs
        self._requote = requote

    def __call__(self, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        if not isinstance(val, str):
            raise TypeError("Argument should be str")
        if not val:
            return ""
        bval = cast(str, val).encode("utf8", errors="ignore")
        ret = bytearray()
        pct = bytearray()
        safe = self._safe
        safe += ALLOWED
        if not self._qs:
            safe += "+&=;"
        safe += self._protected
        bsafe = safe.encode("ascii")
        idx = 0
        while idx < len(bval):
            ch = bval[idx]
            idx += 1

            if pct:
                if ch in BASCII_LOWERCASE:
                    ch = ch - 32  # convert to uppercase
                pct.append(ch)
                if len(pct) == 3:  # pragma: no branch   # peephole optimizer
                    buf = pct[1:]
                    if not _IS_HEX.match(buf):
                        ret.extend(b"%25")
                        pct.clear()
                        idx -= 2
                        continue
                    try:
                        unquoted = chr(int(pct[1:].decode("ascii"), base=16))
                    except ValueError:
                        ret.extend(b"%25")
                        pct.clear()
                        idx -= 2
                        continue

                    if unquoted in self._protected:
                        ret.extend(pct)
                    elif unquoted in safe:
                        ret.append(ord(unquoted))
                    else:
                        ret.extend(pct)
                    pct.clear()

                # special case, if we have only one char after "%"
                elif len(pct) == 2 and idx == len(bval):
                    ret.extend(b"%25")
                    pct.clear()
                    idx -= 1

                continue

            elif ch == ord("%") and self._requote:
                pct.clear()
                pct.append(ch)

                # special case if "%" is last char
                if idx == len(bval):
                    ret.extend(b"%25")

                continue

            if self._qs:
                if ch == ord(" "):
                    ret.append(ord("+"))
                    continue
            if ch in bsafe:
                ret.append(ch)
                continue

            ret.extend((f"%{ch:02X}").encode("ascii"))

        ret2 = ret.decode("ascii")
        if ret2 == val:
            return val
        return ret2


class _Unquoter:
    def __init__(self, *, unsafe: str = "", qs: bool = False) -> None:
        self._unsafe = unsafe
        self._qs = qs
        self._quoter = _Quoter()
        self._qs_quoter = _Quoter(qs=True)

    def __call__(self, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        if not isinstance(val, str):
            raise TypeError("Argument should be str")
        if not val:
            return ""
        decoder = cast(codecs.BufferedIncrementalDecoder, utf8_decoder())
        ret = []
        idx = 0
        while idx < len(val):
            ch = val[idx]
            idx += 1
            if ch == "%" and idx <= len(val) - 2:
                pct = val[idx : idx + 2]
                if _IS_HEX_STR.fullmatch(pct):
                    b = bytes([int(pct, base=16)])
                    idx += 2
                    try:
                        unquoted = decoder.decode(b)
                    except UnicodeDecodeError:
                        start_pct = idx - 3 - len(decoder.buffer) * 3
                        ret.append(val[start_pct : idx - 3])
                        decoder.reset()
                        try:
                            unquoted = decoder.decode(b)
                        except UnicodeDecodeError:
                            ret.append(val[idx - 3 : idx])
                            continue
                    if not unquoted:
                        continue
                    if self._qs and unquoted in "+=&;":
                        to_add = self._qs_quoter(unquoted)
                        if to_add is None:  # pragma: no cover
                            raise RuntimeError("Cannot quote None")
                        ret.append(to_add)
                    elif unquoted in self._unsafe:
                        to_add = self._quoter(unquoted)
                        if to_add is None:  # pragma: no cover
                            raise RuntimeError("Cannot quote None")
                        ret.append(to_add)
                    else:
                        ret.append(unquoted)
                    continue

            if decoder.buffer:
                start_pct = idx - 1 - len(decoder.buffer) * 3
                ret.append(val[start_pct : idx - 1])
                decoder.reset()

            if ch == "+":
                if not self._qs or ch in self._unsafe:
                    ret.append("+")
                else:
                    ret.append(" ")
                continue

            if ch in self._unsafe:
                ret.append("%")
                h = hex(ord(ch)).upper()[2:]
                for ch in h:
                    ret.append(ch)
                continue

            ret.append(ch)

        if decoder.buffer:
            ret.append(val[-len(decoder.buffer) * 3 :])

        ret2 = "".join(ret)
        if ret2 == val:
            return val
        return ret2
