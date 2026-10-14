from __future__ import annotations

import json
import logging


class JsonFileHandler(logging.FileHandler):
    def format(self, record: logging.LogRecord) -> str:
        json_data = json.loads(record.getMessage())
        record.json_data = json_data
        return json.dumps(json_data, ensure_ascii=False, indent=4)

    def emit(self, record: logging.LogRecord) -> None:
        with open(self.baseFilename, "w", encoding="utf-8") as f:
            f.write(self.format(record))
