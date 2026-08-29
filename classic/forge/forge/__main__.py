import logging
import os

import uvicorn
from dotenv import load_dotenv

from forge.logging.config import configure_logging

logger = logging.getLogger(__name__)

logo = """\n\n
       d8888          888             .d8888b.  8888888b. 88888888888
     d88P888          888            888    888 888    888    888
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888
  d88P   888 888  888 888   888  888 888    888 888           888
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888


                8888888888
                888
                888      .d88b.  888d888 .d88b.   .d88b.
                888888  d88""88b 888P"  d88P"88b d8P  Y8b
                888     888  888 888    888  888 88888888
                888     Y88..88P 888    Y88b 888 Y8b.
                888      "Y88P"  888     "Y88888  "Y8888
                                             888
                                        Y8b d88P
                                         "Y88P"                v0.2.0
\n"""

if __name__ == "__main__":
    print(logo)
    port = os.getenv("PORT", 8000)
    configure_logging()
    logger.info(f"Agent server starting on http://localhost:{port}")
    load_dotenv()

    uvicorn.run(
        "forge.app:app",
        host="localhost",
        port=int(port),
        log_level="error",
        # Reload on changes to code or .env
        reload=True,
        reload_dirs=os.path.dirname(os.path.dirname(__file__)),
        reload_excludes="*.py",  # Cancel default *.py include pattern
        reload_includes=[
            f"{os.path.basename(os.path.dirname(__file__))}/**/*.py",
            ".*",
            ".env",
        ],
    )
