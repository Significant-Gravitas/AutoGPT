from __future__ import annotations
import abc
import datetime
import uuid
from sklearn.cluster import KMeans
import numpy as np
import math
import hashlib
from typing import TYPE_CHECKING, Optional, Union, Literal, Tuple
from langchain_core.embeddings import Embeddings
from langchain.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain.vectorstores.chroma import Chroma

from pydantic import BaseModel, Field, validator
from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)

from enum import Enum

class DocumentType(str, Enum):
    TASK = "task"
    DOCUMENTS = "documents"
    MESSAGE_AGENT_USER = "message_agent_user"
    ALL = '*'

class FilterType(str, Enum):
    EQUAL = "$eq"
    REGEX = "$regex"
    START_WITH ="$startswith"
    IN = "$in"

class Filter(BaseModel):
    filter_type : FilterType
    value : str


Filter.update_forward_refs()


class RetrievedDocument(Document) :
    similarity_score : float
    vector_store_wrapper : VectorStoreWrapper
    embeddings : Optional[Embeddings] = None

    class Config(Document.Config):
        arbitrary_types_allowed = True


    def __init__(self, vector_store_wrapper : VectorStoreWrapper ,  document = Document, similarity_score = float):
        super().__init__(**document.dict(), similarity_score=similarity_score , vector_store_wrapper = vector_store_wrapper)

    @staticmethod
    def generate_uuid():
        return 'DOC' + str(uuid.uuid4())
    # async def get_embeddings(self):
    #     return await self.vector_store_wrapper._get_embeding_from_document_id( id = self.metadata["document_id"])

class SearchFilter(BaseModel):

    filters : dict[str, Filter]

    @validator('filters', always=True)
    def check_agent_or_user_id(cls, v, values, **kwargs):
        if 'agent_id' not in v.keys() and 'user_id' not in v.keys() and 'plan_id' not in v.keys():
            raise ValueError('Either plan_id, agent_id or user_id must be provided')

        if 'type' in v.keys():
            raise ValueError('type is a special value and can not be used as a filter')

        return v

    def add_filter(self, key: str, filter: Filter):
        if not isinstance(filter, Filter):
            raise ValueError(f'Filter {key} is not a valid filter')
        if key == 'type':
            raise ValueError('type is a special value and can not be used as a filter')
        self.filters[key] = filter

    # def _check_special_key(self, key: str) -> bool:
    #     if key == 'type':
    #         raise ValueError('type is a special value and can not be used as a filter')
    #     return True


SearchFilter.update_forward_refs()

class VectorStoreWrapper(abc.ABC):
    def __init__(self, vector_store: VectorStore, embedding_model: Embeddings):
        self.vector_store = vector_store
        self.embedding_model = embedding_model


    async def add_document(self, document_type : DocumentType , document : Document , document_id : str = RetrievedDocument.generate_uuid() ) -> str:

        if not any(key in document.metadata for key in ['plan_id', 'agent_id', 'user_id']):
                raise ValueError("At least one of 'plan_id', 'agent_id', or 'user_id' must be provided")

        document.metadata["type"] = document_type.value
        document.metadata["created_at"] = str(datetime.datetime.now())
        document.metadata["document_id"] = document_id
        document.metadata["checksum"] = hashlib.md5(document.page_content.encode()).hexdigest()
        new_ids = await self._add_document(document = document , doc_id = document_id)
        LOG.notice(f"Document added to vector store with id {new_ids[0]}")

        return document_id

    async def search_from_uri(self,
                              query: str,
                              uri: str,
                              search_filters: SearchFilter,
                              nb_results: int = 5
                              ) -> list[RetrievedDocument]:
        # cf : self.generate_filters()
        #search_filters.add_filter('type', Filter(filter_type=FilterType.EQUAL, value=DocumentType.DOCUMENTS.value))
        search_filters.add_filter('source', Filter(filter_type=FilterType.START_WITH, value=uri))
        documents = await  self.get_related_documents(
                                            query= query,
                                            nb_results=nb_results,
                                            search_filters=search_filters,
                                            document_type=DocumentType.DOCUMENTS
                                            )

        return sorted(documents, key=lambda x: x.metadata['created_at'], reverse=True)

    async def get_related_documents(self,
                                    query: str,
                                    nb_results: int,
                                    search_filters: SearchFilter,
                                    document_type: Union[DocumentType, list[DocumentType]],
                                    cluster_search=True,
                                    similarity_threshold=0.8,
                                    similatity_prevalence=0.5,
                                    ) -> list[RetrievedDocument]:
        #self.used_default_filter = False

        k = math.ceil(nb_results * 2.5)  # Always round up
        query_filter = self.generate_filters(filters= search_filters, document_type=document_type)

        try :
            documents = await self.get_documents_with_embeddings(
                query = query,
                k=k,
                include_metadata=True,
                score_threshold = similarity_threshold,
                filter= query_filter,
                )
        except Exception as e :
        #     if (self.used_default_filter) :
        #         LOG.error("This is most likely due because your vector store is not yet supported, help us and implement `VectoreStoreWrapper._make_filter()` for your VectorStore.")
            LOG.error("Error while searching for information, the agent will continue to work with limited capacities")
            return []

        if len(documents) < k:
            return documents[:nb_results]

        if cluster_search:
            sorted_documents = self._sort(documents = documents)
            half_nb_results = math.ceil(nb_results * similatity_prevalence)  # Round up division
            top_similar_documents = sorted_documents[:half_nb_results]
            remaining_results = nb_results - len(top_similar_documents)
            remaining_documents = [doc for doc in sorted_documents if doc not in top_similar_documents]
            #clustered_documents = self._get_most_similar_document_each_cluster(remaining_results, remaining_documents)
            clustered_documents = self._get_centrermost_document_from_each_cluster(remaining_results, remaining_documents)
            return top_similar_documents + clustered_documents

        return documents[:nb_results]

    def _get_centrermost_document_from_each_cluster(self, nb_clusters : int, documents : list[RetrievedDocument]) -> list[RetrievedDocument]:
        vectors = [doc.embeddings for doc in documents]
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42).fit(vectors)

        centermost_document = []
        for cluster_index in range(nb_clusters):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[cluster_index], axis=1)
            closest_index = np.argmin(distances)

            centermost_document.append(documents[closest_index])

        return centermost_document

    def _get_most_similar_document_each_cluster(self, nb_clusters : int, documents : list[RetrievedDocument]) -> list[RetrievedDocument]:
        vectors = [doc.embeddings for doc in documents]
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42).fit(vectors)

        # Create a list to hold the best document per cluster
        best_documents = []

        # Loop through each cluster
        for cluster_index in range(nb_clusters):
            cluster_members_indices = np.where(kmeans.labels_ == cluster_index)[0]
            cluster_members = [documents[i] for i in cluster_members_indices]

            # Find the most similar document within this cluster
            most_similar_document = max(cluster_members, key=lambda doc: doc.similarity_score)
            best_documents.append(most_similar_document)

        return best_documents[:nb_clusters]

    def determine_optimal_clusters(self, embeddings, max_clusters=10):
        """
        Determine the optimal number of clusters for KMeans clustering using the elbow method.

        :param embeddings: A list of document embeddings.
        :param max_clusters: Maximum number of clusters to consider.

        :return: Optimal number of clusters.
        """
        inertia_values = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(embeddings)
            inertia_values.append(kmeans.inertia_)

        # Determine the elbow point, which is the optimal number of clusters
        optimal_clusters = self._find_elbow_point(inertia_values)
        return optimal_clusters

    @staticmethod
    def _find_elbow_point(inertia_values):
        """
        Find the elbow point in the KMeans inertia values, which indicates the optimal number of clusters.

        :param inertia_values: A list of inertia values for different numbers of clusters.

        :return: The number of clusters at the elbow point.
        """
        # This is a simplified way to find the elbow point.
        # For a more accurate determination, consider using more advanced methods.
        n_points = len(inertia_values)
        all_coords = np.vstack((range(1, n_points + 1), inertia_values)).T
        first_point = all_coords[0]
        last_point = all_coords[-1]
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coords - first_point
        scalar_product = np.sum(vec_from_first * np.matlib.repmat(line_vec_norm, n_points, 1), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        index_of_best_point = np.argmax(dist_to_line)

        return index_of_best_point + 1

    def generate_filters(self, filters : SearchFilter,  document_type: Union[DocumentType, list[DocumentType]]) -> dict:
        filter = {}
        for key, value in filters.filters.items():
            if not isinstance(value, Filter):
                raise ValueError(f'Filter {key} is not a valid filter')
            filter.update(self._make_filter(key, value))

        if isinstance(document_type, list):
            if DocumentType.ALL not in document_type:
                filter.update(self._make_filter('type', Filter(filter_type=FilterType.IN, value=document_type.value)))
            else :
                raise ValueError("ALL can not be used with other types")
        else:
            if document_type != DocumentType.ALL:
                filter.update(self._make_filter('type', Filter(filter_type=FilterType.EQUAL, value=document_type.value)))

        return self._compile_filter(filter)

    async def get_documents(self,
                      query: str,
                      k: int,
                      include_metadata: bool ,
                      score_threshold: float,
                      filter: dict) -> list[RetrievedDocument]:
        LOG.warning("Vector Store not supported... Attempting fallback...")
        documents = await self.vector_store.asimilarity_search_by_vector(
            query,
            k=k,
            include_metadata=include_metadata,
            filter= filter,
        )
        return [doc for doc in documents if doc.similarity_score >= score_threshold]

    @abc.abstractmethod
    async def get_documents_with_embeddings(self,
                      query: str,
                      k: int,
                      include_metadata: bool ,
                      score_threshold: float,
                      filter: dict) -> list[RetrievedDocument]:
        ...

    @abc.abstractmethod
    async def _add_document(self, document : Document, doc_id : str):
        ...

    @abc.abstractmethod
    def _make_filter(self, key: str, filter: Filter) -> dict:
        ...

    @abc.abstractmethod
    def _compile_filter(self, filters: dict) -> dict:
        ...

    @abc.abstractmethod
    async def _get_document_by_id(self, vector_store : VectorStore , id : str):
        ...

    @abc.abstractmethod
    def _sort(self, documents : list[RetrievedDocument]) -> list[RetrievedDocument]:
        ...


    @abc.abstractmethod
    async def _get_embeddings_from_document_id(self, id : str):
        ...


    async def _get_documents_with_embeddings_fallback_method(self,
                      query: str,
                      k: int,
                      include_metadata: bool ,
                      score_threshold: float,
                      filter: dict
                      ) -> list[RetrievedDocument]:

        documents = await self.get_documents(
            query=query,
            k=k,
            include_metadata=include_metadata,
            score_threshold=score_threshold,
            filter=filter,
        )
        for doc in documents:
            doc.embeddings = await self._get_embeddings_from_document_id(id = doc.metadata["document_id"])

        return documents


RetrievedDocument.update_forward_refs()


class ChromaWrapper(VectorStoreWrapper):

    vector_store : Chroma

    async def _add_document(self, document : Document, doc_id : str):
        return await self.vector_store.aadd_documents(documents = [document], ids = [doc_id])

    def _make_filter(self, key: str, filter: Filter) -> dict:
        return {key: {filter.filter_type.value: filter.value}}

    def _compile_filter(self, filters: dict) -> dict:
        if len(filters) > 1:
            filters = {"$and": [ {key : filter} for key , filter in  filters.items()]}
        return filters

    async def _get_document_by_id(self, id : str):
        return self.vector_store._collection.get(ids=[id] , include=["embeddings", "metadatas", "documents"])

    async def _get_embeddings_from_document_id(self, id : str):
        document = await self._get_document_by_id(id = id)
        return document['embeddings'][0]

    async def get_documents(self,
                      query: str,
                      k: int,
                      include_metadata: bool ,
                      score_threshold: float,
                      filter: dict) -> list[RetrievedDocument]:
        documents = await self.vector_store.asimilarity_search_with_relevance_scores(
            query,
            k=k,
            filter= filter,
            #include_metadata=include_metadata,
            #include_vector=True,
        )
        LOG.notice("Score threshold not yet implemented... Starting fallback method...")
        return [RetrievedDocument(vector_store_wrapper=self, document=doc[0] , similarity_score = doc[1]) for doc in documents if doc[1] >= score_threshold]

    async def get_documents_with_embeddings(self,
                      query: str,
                      k: int,
                      include_metadata: bool ,
                      score_threshold: float,
                      filter: dict
                      ) -> list[RetrievedDocument]:

        # documents = await self.get_documents(
        #     query=query,
        #     k=k,
        #     include_metadata=include_metadata,
        #     score_threshold=score_threshold,
        #     filter=filter,
        # )
        # for doc in documents:
        #     doc.embeddings = await self._get_embeddings_from_document_id(id = doc.metadata["document_id"])

        return await self._get_documents_with_embeddings_fallback_method(
            query=query,
            k=k,
            include_metadata=include_metadata,
            score_threshold=score_threshold,
            filter=filter,
        )

    def _sort(self, documents : list[RetrievedDocument]) -> list[RetrievedDocument]:
            # if (isinstance(self.vector_store, Chroma)) :
        return documents
            # else :
            #     #Workaround for non supported vector stores
            #     return sorted(documents, key=lambda r: r.similarity_score, reverse=True)


# """
# from langchain.text_splitter import (
#     Language,
#     RecursiveCharacterTextSplitter,
# )
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings

# text_splitter = SemanticChunker(OpenAIEmbeddings())

# """
# 1. Load
# 2. Split
# 3. Add to vector store
# 4. Search
# 5. Retrieve
# """
