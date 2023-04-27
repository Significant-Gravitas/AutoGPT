"""Implementation of memory backend using Vectara (https://vectara.com)"""
import hashlib
import json
import logging
import time

import requests
from colorama import Fore, Style

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.memory.base import MemoryProviderSingleton


class JWTTokenProvider:
    """Provides OAuth2 JWT tokens for authorization."""

    def __init__(self, customer_id, app_id, app_secret):
        self._app_id = app_id
        self._app_secret = app_secret
        self._token_endpoint = f"https://vectara-prod-{customer_id}.auth.us-west-2.amazoncognito.com/oauth2/token"
        self._token = None
        self._token_refresh_ts = 0

    def get_jwt_token(self):
        """Returns JWT that can be attached for authorization."""
        from authlib.integrations.requests_client import OAuth2Session

        # Refresh token every 5 mins.
        if (time.time() - self._token_refresh_ts) < 5 * 60:
            return self._token

        logging.debug(
            "Refreshing JWT token after %s secs.", time.time() - self._token_refresh_ts
        )
        session = OAuth2Session(self._app_id, self._app_secret, scope="")
        token = session.fetch_token(
            self._token_endpoint, grant_type="client_credentials"
        )  # authorization_code

        self._token_refresh_ts = time.time()
        self._token = token["access_token"]
        return self._token


class VectaraMemory(MemoryProviderSingleton):
    """A memory provider backed by Vectara (https://vectara.com)."""

    def __init__(self, cfg):
        self._jwt_provider = JWTTokenProvider(
            cfg.vectara_customer_id, cfg.vectara_app_id, cfg.vectara_app_secret
        )
        self._customer_id = cfg.vectara_customer_id
        self._corpus_id = self.get_corpus_id(cfg.memory_index)
        logging.debug("Using corpus id %s", self._corpus_id)

    def _get_post_headers(self):
        """Returns headers that should be attached to each post request."""
        return {
            "Authorization": f"Bearer {self._jwt_provider.get_jwt_token()}",
            "customer-id": str(self._customer_id),
            "Content-Type": "application/json",
        }

    def get_corpus_id(self, index_name):
        """Returns the corpus id for the passed 'index_name'.

        Vectara APIs use int corpus (aka index) idss instead of string names. This
        method can be used to get the int corpus id.
        If a corpus with the name already exists, its id is returned. Otherwise, a new
        corpus is created with the name, and its id is returned.
        """
        response = requests.post(
            url="https://api.vectara.io/v1/list-corpora",
            headers=self._get_post_headers(),
            data=json.dumps({"numResults": 2, "filter": f"^{index_name}$"}),
            timeout=15,
        )
        if response.status_code != 200:
            logger.typewriter_log(
                "FAILED TO LIST VECTARA CORPORA",
                Fore.RED,
                Style.BRIGHT
                + (
                    f"code {response.status_code}, reason {response.reason}, details {response.text}"
                )
                + Style.RESET_ALL,
            )
            logger.double_check(
                "Please ensure you have setup and configured Vectara properly for use."
                f"You can check out {Fore.CYAN + Style.BRIGHT}"
                "https://github.com/Torantulino/Auto-GPT#-vectara-setup"
                f"{Style.RESET_ALL} to ensure you've set up everything correctly."
            )
            exit(1)
        result = response.json()

        if result["status"]:
            logger.typewriter_log(
                "LIST VECTARA CORPORA CALL RETURNED ERROR. CONTACT support@vectara.com FOR HELP.",
                Fore.RED,
                Style.BRIGHT + json.dumps(result["status"]) + Style.RESET_ALL,
            )
            exit(1)
        corpora = result["corpus"]
        if len(corpora) == 1:
            return corpora[0]["id"]
        if len(corpora) > 1:
            logger.typewriter_log(
                "MULTIPLE VECTARA CORPORA FOUND FOR AUTO-GPT",
                Fore.RED,
                Style.BRIGHT
                + (
                    f"Multiple {len(corpora)} corpora (aka index) found "
                    f"with name {index_name}. Please delete uneeded corpora, "
                    "or use a different corpus name (MEMORY_INDEX) for Auto-GPT."
                )
                + Style.RESET_ALL,
            )
            exit(1)

        # Auto-GPT corpus doesn't exist. Create it.
        response = requests.post(
            url="https://api.vectara.io/v1/create-corpus",
            headers=self._get_post_headers(),
            data=json.dumps(
                {
                    "corpus": {
                        "name": index_name,
                        "description": "Set of documents for Auto-GPT.",
                    }
                }
            ),
            timeout=15,
        )

        if response.status_code != 200:
            logger.typewriter_log(
                "VECTARA CORPUS CREATION FAILED",
                Fore.RED,
                Style.BRIGHT
                + (
                    f"Create Corpus failed with code {response.status_code}, "
                    f"reason {response.reason}, text {response.text}. "
                    "Contact support@vectara.com for help"
                )
                + Style.RESET_ALL,
            )
            exit(1)
        result = response.json()

        if result["status"] and result["status"]["code"] != "OK":
            logger.typewriter_log(
                "VECTARA CORPUS CREATION FAILED. CONTACT support@vectara.com FOR HELP.",
                Fore.RED,
                Style.BRIGHT + json.dumps(result["status"]) + Style.RESET_ALL,
            )
            exit(1)

        return result["corpusId"]

    def add(self, data):
        """Adds 'data' to the Vectara index."""
        doc_id = hashlib.sha256(data.encode("utf8")).hexdigest()
        request = {}
        request["customer_id"] = self._customer_id
        request["corpus_id"] = self._corpus_id
        request["document"] = {"document_id": doc_id, "parts": [{"text": data}]}

        logging.debug("Sending request %s", json.dumps(request))
        response = requests.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/core/index",
            data=json.dumps(request),
            timeout=30,
        )

        if response.status_code != 200:
            error = (
                f"REST upload failed with code {response.status_code}, "
                f"reason {response.reason}, text {response.text}"
            )
            logging.warning(error)
            return error
        result = response.json()

        allowed_codes = ["OK", "ALREADY_EXISTS"]
        if result["status"] and result["status"]["code"] not in allowed_codes:
            logger.typewriter_log(
                "VECTARA CORPUS INDEX FAILED. CONTACT support@vectara.com FOR HELP.",
                Fore.RED,
                Style.BRIGHT + json.dumps(result["status"]) + Style.RESET_ALL,
            )
            return f"Indexing failed: {json.dumps(result['status'])}"
        return f"Inserting data into memory at uuid: {doc_id}:\n data: {data}"

    def get(self, data):
        """Returns the first result relevant to 'data'."""
        return self.get_relevant(data, 1)

    def clear(self):
        """Clears the index (remove all data in it)."""
        response = requests.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/reset-corpus",
            data=json.dumps(
                {"customer_id": self._customer_id, "corpus_id": self._corpus_id}
            ),
            timeout=60,
        )

        if response.status_code != 200:
            error = (
                f"Reset Corpus failed with code {response.status_code}, "
                f"reason {response.reason}, text {response.text}"
            )
            logging.warning(error)
            return error
        return "Obliterated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        response = requests.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=json.dumps(
                {
                    "query": [
                        {
                            "query": data,
                            "num_results": num_relevant,
                            "corpus_key": [
                                {
                                    "customer_id": self._customer_id,
                                    "corpus_id": self._corpus_id,
                                }
                            ],
                            "reranking_config": {"rerankerId": 272725717},
                        }
                    ]
                }
            ),
            timeout=10,
        )

        if response.status_code != 200:
            logging.error(
                "Query failed with code %d, reason %s, text %s",
                response.status_code,
                response.reason,
                response.text,
            )
            return []

        result = response.json()
        # ignoring status errors
        return [x["text"] for x in result["responseSet"][0]["response"]]

    def get_stats(self):
        return (
            "get_stats not supported. View your index stats at "
            f"https://console.vectara.com/console/corpus/{self._corpus_id}/overview"
        )


# For manually testing Vectara memory backend.
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG
    )
    config = Config()
    # Set before running.
    config.vectara_customer_id = 123
    config.vectara_app_id = "92d..."  # Corpus Admin
    config.vectara_app_secret = "3a4..."
    config.memory_index = "Test-Vectara-AutoGpt-Integration"

    memory = VectaraMemory(config)

    memory.add("The new avatar movie is out.")
    memory.add("Cricket is a long but interesting game.")
    memory.add("Hiking keeps one healthy.")
    result = memory.get("films")
    assert len(result) == 1
    assert (
        result[0] == "The new avatar movie is out."
    ), f"Result should be 'The new avatar movie is out.', found '{result[0]}'"
