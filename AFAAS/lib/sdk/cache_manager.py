from __future__ import annotations

import json
import os
import time

import redis
from dotenv import load_dotenv


class CacheManager:
    def __init__(self):
        load_dotenv()
        use_redis = os.getenv("USE_REDIS", "False").lower() == "true"
        self.redis_config = {
            "host": os.getenv("REDIS_HOST"),
            "port": int(os.getenv("REDIS_PORT")),
            "db": int(os.getenv("REDIS_DB")),
        }
        self.json_file_path = os.getenv("JSON_CACHE_FILE_PATH")
        self.cache = {}

        self.use_redis = False
        if use_redis:
            try:
                self.redis_client = redis.Redis(**self.redis_config)
                self.use_redis = True
            except:
                self.load_from_json()
        else:
            self.load_from_json()

    def load_from_json(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as file:
                self.cache = json.load(file)

    def save_to_json(self):
        # Extract the directory from the file path
        directory = os.path.dirname(self.json_file_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        with open(self.json_file_path, "w") as file:
            json.dump(self.cache, file, indent=4)

    def get(self, key):
        if self.use_redis:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        return self.cache.get(key)

    def set(self, key, value):
        if self.use_redis:
            self.redis_client.set(key, json.dumps(value))
        else:
            self.cache[key] = value
            self.save_to_json()

    def clear_cache(self):
        """
        Clear the entire cache.
        """
        if self.use_redis:
            self.redis_client.flushdb()  # Clears all keys in the current Redis database
        else:
            self.cache = {}  # Reset the in-memory cache
            self.save_to_json()  # Save the empty cache to the JSON file

    def set_cache_time(self, time_stamp=time.time()):
        """Set the last updated time in the cache."""
        self.set("last_updated", time_stamp)

    def get_cache_time(self):
        """Get the last updated time from the cache. Returns 0 if not set."""
        return self.get("last_updated") or 0

    def get_all_categories(self) -> list[str]:
        """
        Retrieve all categories from the cache, excluding special keys like 'last_updated'.

        Returns:
            list[str]: A list of all category names stored in the cache.
        """
        if self.use_redis:
            # Retrieve all keys from Redis and filter out 'last_updated'
            all_keys = self.redis_client.keys("*")
            all_categories = [
                key.decode("utf-8")
                for key in all_keys
                if key.decode("utf-8") != "last_updated"
            ]
        else:
            # For JSON cache, filter out 'last_updated' from the keys
            all_categories = list(set(self.cache.keys()) - {"last_updated"})

        return all_categories
