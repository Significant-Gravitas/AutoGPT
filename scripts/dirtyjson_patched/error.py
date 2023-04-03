"""JSON decode error class
"""

__all__ = ['Error']


class Error(ValueError):
    """Subclass of ValueError with the following additional properties:

    msg: The unformatted error message
    doc: The JSON document being parsed
    pos: The start index of doc where parsing failed
    end: The end index of doc where parsing failed (may be None)
    lineno: The line corresponding to pos
    colno: The column corresponding to pos
    endlineno: The line corresponding to end (may be None)
    endcolno: The column corresponding to end (may be None)

    """
    # Note that this exception is used from _speedups
    def __init__(self, msg, doc, pos):
        ValueError.__init__(self, errmsg(msg, doc, pos))
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno, self.colno = linecol(doc, pos)


def linecol(doc, pos):
    lineno = doc.count('\n', 0, pos) + 1
    if lineno == 1:
        colno = pos + 1
    else:
        colno = pos - doc.rindex('\n', 0, pos)
    return lineno, colno


def errmsg(msg, doc, pos):
    lineno, colno = linecol(doc, pos)
    msg = msg.replace('%r', repr(doc[pos:pos + 1]))
    fmt = '%s: line %d column %d (char %d)'
    return fmt % (msg, lineno, colno, pos)
