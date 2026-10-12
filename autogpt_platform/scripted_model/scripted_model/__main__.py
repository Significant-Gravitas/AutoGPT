"""``python -m scripted_model``: serve scripts from SCRIPTED_MODEL_SCRIPTS_DIR."""

import logging
import os
from pathlib import Path

from .server import ScriptedModelServer, logger


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("SCRIPTED_MODEL_LOG_LEVEL", "INFO"),
        format="%(asctime)s scripted-model %(levelname)s %(message)s",
    )
    port = int(os.environ.get("SCRIPTED_MODEL_PORT", "4010"))
    scripts_dir = Path(os.environ.get("SCRIPTED_MODEL_SCRIPTS_DIR", "/scripts"))
    server = ScriptedModelServer(("0.0.0.0", port), scripts_dir)
    known = sorted(p.stem for p in scripts_dir.glob("*.json"))
    logger.info("serving on :%s with scripts from %s: %s", port, scripts_dir, known)
    server.serve_forever()


if __name__ == "__main__":
    main()
