import atexit
import os
import json
from typing import Literal
from annoy import AnnoyIndex
from memory.base import MemoryProviderSingleton, get_ada_embedding

class AnnoyMemory(MemoryProviderSingleton):
    """
    AnnoyMemory is a memory provider singleton that leverages the Annoy library to create an index
    for efficient nearest neighbor search.
    """
    def __init__(self, cfg):
        """
        Initialize the AnnoyMemory class.
        
        :param cfg: A configuration object containing the index file and metadata file paths.
        """
        self.dimension = 1536
        self.metric: Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot'] = 'angular'
        self.index_file = cfg.annoy_index_file
        self.metadata_file = cfg.annoy_metadata_file

        self.index = AnnoyIndex(self.dimension, metric=self.metric)

        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
             # Load existing index into a temporary index
            temp_index = AnnoyIndex(self.dimension, metric=self.metric)
            temp_index.load(self.index_file)

            # Clone items from temporary index to the new index
            for i in range(temp_index.get_n_items()):
                self.index.add_item(i, temp_index.get_item_vector(i))

            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

            # Ensure the directory for the index file and metadata file exists
            index_dir = os.path.dirname(self.index_file)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
            self.ensure_directory_exists_and_save_metadata()

            # Register save_on_exit function to be called when the program is closing
            atexit.register(self.save_on_exit)

            # Remove the dummy vector from the metadata
            self.metadata.append({"raw_text": "DUMMY"})
            self.metadata.pop()

            # Save the metadata file
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)

    def save_on_exit(self):
        """
        Save the Annoy index and metadata on program exit.
        """
        self.index.build(10)
        self.index.save(self.index_file)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
        print("Annoy index and metadata saved on exit.")

    def ensure_directory_exists_and_save_metadata(self):
        """
        Ensure the directories for the metadata file exist, and save the metadata.
        """
        metadata_dir = os.path.dirname(self.metadata_file)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)

    def add(self, data):
        """
        Add a data point to the Annoy index and update the metadata.
        
        :param data: The data point to add.
        :return: A text string describing the insertion of the data point.
        """
        vector = get_ada_embedding(data)
        vec_num = len(self.metadata)
        self.metadata.append({"raw_text": data})
        self.index.add_item(vec_num, vector)
        self.ensure_directory_exists_and_save_metadata()
        _text = f"Inserting data into memory at index: {vec_num}:\n data: {data}"
        return _text

    def get(self, data):
        """
        Get the most relevant data point from the memory given an input data point.
        
        :param data: The input data point.
        :return: The most relevant data point from the memory.
        """
        return self.get_relevant(data, 1)

    def clear(self):
        """
        Remove the index file and metadata file, and reinitialize the index and metadata.
        
        :return: A string indicating the memory has been cleared.
        """
        os.remove(self.index_file)
        os.remove(self.metadata_file)
        self.index = AnnoyIndex(self.dimension, metric=self.metric)
        self.metadata = []
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = get_ada_embedding(data)
        nearest_indices, distances = self.index.get_nns_by_vector(query_embedding, num_relevant, include_distances=True)
        sorted_indices = [i for _, i in sorted(zip(distances, nearest_indices))]
        return [self.metadata[i]["raw_text"] for i in sorted_indices]

    def get_stats(self):
        """
        Get the statistics of the Annoy index.
        
        :return: A dictionary containing the number of items in the index.
        """
        return {"n_items": self.index.get_n_items()}
