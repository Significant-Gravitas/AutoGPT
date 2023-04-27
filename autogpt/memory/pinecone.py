import pinecone
from colorama import Fore, Style

from autogpt.llm_utils import get_ada_embedding
from autogpt.logs import logger
from autogpt.memory.base import MemoryProviderSingleton
from autogpt.utils import clean_input


class PineconeMemory(MemoryProviderSingleton):
    def __init__(self, cfg):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = "auto-gpt"

        try:
            pinecone.whoami()
        except Exception as e:
            logger.typewriter_log(
                "FAILED TO CONNECT TO PINECONE",
                Fore.RED,
                Style.BRIGHT + str(e) + Style.RESET_ALL,
            )
            logger.double_check(
                "Please ensure you have setup and configured Pinecone properly for use."
                + f"You can check out {Fore.CYAN + Style.BRIGHT}"
                "https://github.com/Torantulino/Auto-GPT#-pinecone-api-key-setup"
                f"{Style.RESET_ALL} to ensure you've set up everything correctly."
            )
            exit(1)

        if table_name not in pinecone.list_indexes():
            pinecone.create_index(
                table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )
        self.index = pinecone.Index(table_name)
        # gets stats for index pod, extracts index_fullness, converts to percentage.
        # prompts user to clear or not based on preference; extracts vector count
        # and assigns it to self.vec_num
        # index fullness percentage and vector count are displayed in prompt
        index_stats = self.index.describe_index_stats()
        namespaces = index_stats.get("namespaces", {})
        default_namespace = namespaces.get("", {})
        self.vec_num = default_namespace.get("vector_count", None)
        if self.vec_num == None:
            self.vec_num = "0"
        index_fullness = index_stats.index_fullness * 100

        if index_fullness == 90:
            fullness_percent = f"{Fore.YELLOW}{index_fullness}%{Style.RESET_ALL}"
        elif index_fullness < 10:
            fullness_percent = f"{Fore.GREEN}<10%{Style.RESET_ALL}"
        else:
            fullness_percent = f"{Fore.GREEN}{index_fullness}%{Style.RESET_ALL}"

        should_clear = clean_input(
            f"""{Fore.CYAN}Memory storage:{Style.RESET_ALL} Your Pinecone index pod is currently at 
                {fullness_percent} of capacity, with {self.vec_num} vector(s). Clear the index?
                {Fore.MAGENTA}(y/n): {Style.RESET_ALL}"""
        )

        if should_clear.lower() == "y":
            self.index.delete(delete_all=True)
            print(
                f"""
{Fore.GREEN}Pinecone index cleared.{Style.RESET_ALL}"""
            )
            return None
        elif should_clear.lower() == "n":
            pass

    def add(self, data):
        vector = get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = get_ada_embedding(data)
        results = self.index.query(
            query_embedding, top_k=num_relevant, include_metadata=True
        )
        sorted_results = sorted(results.matches, key=lambda x: x.score)
        return [str(item["metadata"]["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()
