import io
from typing import Any, Iterable, List, Optional
from urllib.parse import urlencode

from multidict import MultiDict, MultiDictProxy

from . import hdrs, multipart, payload
from .helpers import guess_filename
from .payload import Payload

__all__ = ("FormData",)


class FormData:
    """Helper class for form body generation.

    Supports multipart/form-data and application/x-www-form-urlencoded.
    """

    def __init__(
        self,
        fields: Iterable[Any] = (),
        quote_fields: bool = True,
        charset: Optional[str] = None,
    ) -> None:
        self._writer = multipart.MultipartWriter("form-data")
        self._fields: List[Any] = []
        self._is_multipart = False
        self._is_processed = False
        self._quote_fields = quote_fields
        self._charset = charset

        if isinstance(fields, dict):
            fields = list(fields.items())
        elif not isinstance(fields, (list, tuple)):
            fields = (fields,)
        self.add_fields(*fields)

    @property
    def is_multipart(self) -> bool:
        return self._is_multipart

    def add_field(
        self,
        name: str,
        value: Any,
        *,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
        content_transfer_encoding: Optional[str] = None,
    ) -> None:

        if isinstance(value, io.IOBase):
            self._is_multipart = True
        elif isinstance(value, (bytes, bytearray, memoryview)):
            if filename is None and content_transfer_encoding is None:
                filename = name

        type_options: MultiDict[str] = MultiDict({"name": name})
        if filename is not None and not isinstance(filename, str):
            raise TypeError(
                "filename must be an instance of str. " "Got: %s" % filename
            )
        if filename is None and isinstance(value, io.IOBase):
            filename = guess_filename(value, name)
        if filename is not None:
            type_options["filename"] = filename
            self._is_multipart = True

        headers = {}
        if content_type is not None:
            if not isinstance(content_type, str):
                raise TypeError(
                    "content_type must be an instance of str. " "Got: %s" % content_type
                )
            headers[hdrs.CONTENT_TYPE] = content_type
            self._is_multipart = True
        if content_transfer_encoding is not None:
            if not isinstance(content_transfer_encoding, str):
                raise TypeError(
                    "content_transfer_encoding must be an instance"
                    " of str. Got: %s" % content_transfer_encoding
                )
            headers[hdrs.CONTENT_TRANSFER_ENCODING] = content_transfer_encoding
            self._is_multipart = True

        self._fields.append((type_options, headers, value))

    def add_fields(self, *fields: Any) -> None:
        to_add = list(fields)

        while to_add:
            rec = to_add.pop(0)

            if isinstance(rec, io.IOBase):
                k = guess_filename(rec, "unknown")
                self.add_field(k, rec)  # type: ignore[arg-type]

            elif isinstance(rec, (MultiDictProxy, MultiDict)):
                to_add.extend(rec.items())

            elif isinstance(rec, (list, tuple)) and len(rec) == 2:
                k, fp = rec
                self.add_field(k, fp)  # type: ignore[arg-type]

            else:
                raise TypeError(
                    "Only io.IOBase, multidict and (name, file) "
                    "pairs allowed, use .add_field() for passing "
                    "more complex parameters, got {!r}".format(rec)
                )

    def _gen_form_urlencoded(self) -> payload.BytesPayload:
        # form data (x-www-form-urlencoded)
        data = []
        for type_options, _, value in self._fields:
            data.append((type_options["name"], value))

        charset = self._charset if self._charset is not None else "utf-8"

        if charset == "utf-8":
            content_type = "application/x-www-form-urlencoded"
        else:
            content_type = "application/x-www-form-urlencoded; " "charset=%s" % charset

        return payload.BytesPayload(
            urlencode(data, doseq=True, encoding=charset).encode(),
            content_type=content_type,
        )

    def _gen_form_data(self) -> multipart.MultipartWriter:
        """Encode a list of fields using the multipart/form-data MIME format"""
        if self._is_processed:
            raise RuntimeError("Form data has been processed already")
        for dispparams, headers, value in self._fields:
            try:
                if hdrs.CONTENT_TYPE in headers:
                    part = payload.get_payload(
                        value,
                        content_type=headers[hdrs.CONTENT_TYPE],
                        headers=headers,
                        encoding=self._charset,
                    )
                else:
                    part = payload.get_payload(
                        value, headers=headers, encoding=self._charset
                    )
            except Exception as exc:
                raise TypeError(
                    "Can not serialize value type: %r\n "
                    "headers: %r\n value: %r" % (type(value), headers, value)
                ) from exc

            if dispparams:
                part.set_content_disposition(
                    "form-data", quote_fields=self._quote_fields, **dispparams
                )
                # FIXME cgi.FieldStorage doesn't likes body parts with
                # Content-Length which were sent via chunked transfer encoding
                assert part.headers is not None
                part.headers.popall(hdrs.CONTENT_LENGTH, None)

            self._writer.append_payload(part)

        self._is_processed = True
        return self._writer

    def __call__(self) -> Payload:
        if self._is_multipart:
            return self._gen_form_data()
        else:
            return self._gen_form_urlencoded()
