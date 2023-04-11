from ._util import to_string
from .document import Document


class Result:
    """
    Represents the result of a search query, and has an array of Document
    objects
    """

    def __init__(
        self, res, hascontent, duration=0, has_payload=False, with_scores=False
    ):
        """
        - **snippets**: An optional dictionary of the form
        {field: snippet_size} for snippet formatting
        """

        self.total = res[0]
        self.duration = duration
        self.docs = []

        step = 1
        if hascontent:
            step = step + 1
        if has_payload:
            step = step + 1
        if with_scores:
            step = step + 1

        offset = 2 if with_scores else 1

        for i in range(1, len(res), step):
            id = to_string(res[i])
            payload = to_string(res[i + offset]) if has_payload else None
            # fields_offset = 2 if has_payload else 1
            fields_offset = offset + 1 if has_payload else offset
            score = float(res[i + 1]) if with_scores else None

            fields = {}
            if hascontent and res[i + fields_offset] is not None:
                fields = (
                    dict(
                        dict(
                            zip(
                                map(to_string, res[i + fields_offset][::2]),
                                map(to_string, res[i + fields_offset][1::2]),
                            )
                        )
                    )
                    if hascontent
                    else {}
                )
            try:
                del fields["id"]
            except KeyError:
                pass

            try:
                fields["json"] = fields["$"]
                del fields["$"]
            except KeyError:
                pass

            doc = (
                Document(id, score=score, payload=payload, **fields)
                if with_scores
                else Document(id, payload=payload, **fields)
            )
            self.docs.append(doc)

    def __repr__(self):
        return f"Result{{{self.total} total, docs: {self.docs}}}"
