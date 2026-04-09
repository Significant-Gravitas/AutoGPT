from graphiti_core.driver.falkordb import STOPWORDS
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.helpers import validate_group_ids


class AutoGPTFalkorDriver(FalkorDriver):
    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 128,
    ) -> str:
        validate_group_ids(group_ids)

        group_filter = ""
        if group_ids:
            group_filter = f"(@group_id:{'|'.join(group_ids)})"

        sanitized_query = self.sanitize(query)
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word.lower() not in STOPWORDS]
        sanitized_query = " | ".join(filtered_words)

        if not sanitized_query:
            fulltext_query = group_filter
        elif not group_filter:
            fulltext_query = f"({sanitized_query})"
        else:
            fulltext_query = f"{group_filter} ({sanitized_query})"

        if len(fulltext_query) >= max_query_length:
            return ""

        return fulltext_query
