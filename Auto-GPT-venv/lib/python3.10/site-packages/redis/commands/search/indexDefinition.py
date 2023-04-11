from enum import Enum


class IndexType(Enum):
    """Enum of the currently supported index types."""

    HASH = 1
    JSON = 2


class IndexDefinition:
    """IndexDefinition is used to define a index definition for automatic
    indexing on Hash or Json update."""

    def __init__(
        self,
        prefix=[],
        filter=None,
        language_field=None,
        language=None,
        score_field=None,
        score=1.0,
        payload_field=None,
        index_type=None,
    ):
        self.args = []
        self._append_index_type(index_type)
        self._append_prefix(prefix)
        self._append_filter(filter)
        self._append_language(language_field, language)
        self._append_score(score_field, score)
        self._append_payload(payload_field)

    def _append_index_type(self, index_type):
        """Append `ON HASH` or `ON JSON` according to the enum."""
        if index_type is IndexType.HASH:
            self.args.extend(["ON", "HASH"])
        elif index_type is IndexType.JSON:
            self.args.extend(["ON", "JSON"])
        elif index_type is not None:
            raise RuntimeError(f"index_type must be one of {list(IndexType)}")

    def _append_prefix(self, prefix):
        """Append PREFIX."""
        if len(prefix) > 0:
            self.args.append("PREFIX")
            self.args.append(len(prefix))
            for p in prefix:
                self.args.append(p)

    def _append_filter(self, filter):
        """Append FILTER."""
        if filter is not None:
            self.args.append("FILTER")
            self.args.append(filter)

    def _append_language(self, language_field, language):
        """Append LANGUAGE_FIELD and LANGUAGE."""
        if language_field is not None:
            self.args.append("LANGUAGE_FIELD")
            self.args.append(language_field)
        if language is not None:
            self.args.append("LANGUAGE")
            self.args.append(language)

    def _append_score(self, score_field, score):
        """Append SCORE_FIELD and SCORE."""
        if score_field is not None:
            self.args.append("SCORE_FIELD")
            self.args.append(score_field)
        if score is not None:
            self.args.append("SCORE")
            self.args.append(score)

    def _append_payload(self, payload_field):
        """Append PAYLOAD_FIELD."""
        if payload_field is not None:
            self.args.append("PAYLOAD_FIELD")
            self.args.append(payload_field)
