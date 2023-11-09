import json
import json.decoder

from ..errors import StreamParseError


json_decoder = json.JSONDecoder()


def stream_as_text(stream):
    """
    Given a stream of bytes or text, if any of the items in the stream
    are bytes convert them to text.
    This function can be removed once we return text streams
    instead of byte streams.
    """
    for data in stream:
        if not isinstance(data, str):
            data = data.decode('utf-8', 'replace')
        yield data


def json_splitter(buffer):
    """Attempt to parse a json object from a buffer. If there is at least one
    object, return it and the rest of the buffer, otherwise return None.
    """
    buffer = buffer.strip()
    try:
        obj, index = json_decoder.raw_decode(buffer)
        rest = buffer[json.decoder.WHITESPACE.match(buffer, index).end():]
        return obj, rest
    except ValueError:
        return None


def json_stream(stream):
    """Given a stream of text, return a stream of json objects.
    This handles streams which are inconsistently buffered (some entries may
    be newline delimited, and others are not).
    """
    return split_buffer(stream, json_splitter, json_decoder.decode)


def line_splitter(buffer, separator='\n'):
    index = buffer.find(str(separator))
    if index == -1:
        return None
    return buffer[:index + 1], buffer[index + 1:]


def split_buffer(stream, splitter=None, decoder=lambda a: a):
    """Given a generator which yields strings and a splitter function,
    joins all input, splits on the separator and yields each chunk.
    Unlike string.split(), each chunk includes the trailing
    separator, except for the last one if none was found on the end
    of the input.
    """
    splitter = splitter or line_splitter
    buffered = ''

    for data in stream_as_text(stream):
        buffered += data
        while True:
            buffer_split = splitter(buffered)
            if buffer_split is None:
                break

            item, buffered = buffer_split
            yield item

    if buffered:
        try:
            yield decoder(buffered)
        except Exception as e:
            raise StreamParseError(e)
