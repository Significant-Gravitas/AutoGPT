from ._util import to_string


class Suggestion:
    """
    Represents a single suggestion being sent or returned from the
    autocomplete server
    """

    def __init__(self, string, score=1.0, payload=None):
        self.string = to_string(string)
        self.payload = to_string(payload)
        self.score = score

    def __repr__(self):
        return self.string


class SuggestionParser:
    """
    Internal class used to parse results from the `SUGGET` command.
    This needs to consume either 1, 2, or 3 values at a time from
    the return value depending on what objects were requested
    """

    def __init__(self, with_scores, with_payloads, ret):
        self.with_scores = with_scores
        self.with_payloads = with_payloads

        if with_scores and with_payloads:
            self.sugsize = 3
            self._scoreidx = 1
            self._payloadidx = 2
        elif with_scores:
            self.sugsize = 2
            self._scoreidx = 1
        elif with_payloads:
            self.sugsize = 2
            self._payloadidx = 1
        else:
            self.sugsize = 1
            self._scoreidx = -1

        self._sugs = ret

    def __iter__(self):
        for i in range(0, len(self._sugs), self.sugsize):
            ss = self._sugs[i]
            score = float(self._sugs[i + self._scoreidx]) if self.with_scores else 1.0
            payload = self._sugs[i + self._payloadidx] if self.with_payloads else None
            yield Suggestion(ss, score, payload)
