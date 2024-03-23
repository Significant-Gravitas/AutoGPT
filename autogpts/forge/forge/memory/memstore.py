import abc
import hashlib

import chromadb
from chromadb.config import Settings


class MemStore(abc.ABC):
    """
    An abstract class that represents a Memory Store
    """

    @abc.abstractmethod
    def __init__(self, store_path: str):
        """
        Initialize the MemStore with a given store path.

        Args:
            store_path (str): The path to the store.
        """
        pass

    @abc.abstractmethod
    def add_task_memory(self, task_id: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        self.add(collection_name=task_id, document=document, metadatas=metadatas)

    @abc.abstractmethod
    def query_task_memory(
        self,
        task_id: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        """
        Query the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            query (str): The query string.
            filters (dict, optional): The filters to be applied. Defaults to None.
            document_search (dict, optional): The search string. Defaults to None.

        Returns:
            dict: The query results.
        """
        return self.query(
            collection_name=task_id,
            query=query,
            filters=filters,
            document_search=document_search,
        )

    @abc.abstractmethod
    def get_task_memory(
        self, task_id: str, doc_ids: list = None, filters: dict = None
    ) -> dict:
        """
        Get documents from the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list, optional): The IDs of the documents to be retrieved. Defaults to None.
            filters (dict, optional): The filters to be applied. Defaults to None.

        Returns:
            dict: The retrieved documents.
        """
        return self.get(collection_name=task_id, doc_ids=doc_ids, filters=filters)

    @abc.abstractmethod
    def update_task_memory(
        self, task_id: str, doc_ids: list, documents: list, metadatas: list
    ):
        """
        Update documents in the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list): The IDs of the documents to be updated.
            documents (list): The updated documents.
            metadatas (list): The updated metadata.
        """
        self.update(
            collection_name=task_id,
            doc_ids=doc_ids,
            documents=documents,
            metadatas=metadatas,
        )

    @abc.abstractmethod
    def delete_task_memory(self, task_id: str, doc_id: str):
        """
        Delete a document from the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            doc_id (str): The ID of the document to be deleted.
        """
        self.delete(collection_name=task_id, doc_id=doc_id)

    @abc.abstractmethod
    def add(self, collection_name: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the current collection's MemStore.

        Args:
            collection_name (str): The name of the collection.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        pass

    @abc.abstractmethod
    def query(
        self,
        collection_name: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        pass

    @abc.abstractmethod
    def get(
        self, collection_name: str, doc_ids: list = None, filters: dict = None
    ) -> dict:
        pass

    @abc.abstractmethod
    def update(
        self, collection_name: str, doc_ids: list, documents: list, metadatas: list
    ):
        pass

    @abc.abstractmethod
    def delete(self, collection_name: str, doc_id: str):
        pass
