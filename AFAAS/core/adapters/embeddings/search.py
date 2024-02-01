from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from pydantic import Field, validator

from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

#The idea is great but I think we should consider there can be many filters on many keys ; what about "
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
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

class SearchFilter(BaseModel):

    filters : dict[str, Filter]

    @validator('filters', always=True)
    def check_agent_or_user_id(cls, v, values, **kwargs):
        if 'agent_id' not in v.keys() and 'user_id' not in v.keys() and 'plan_id' not in v.keys():
            raise ValueError('Either plan_id, agent_id or user_id must be provided')

        if 'type' in v.keys():
            raise ValueError('type is a special value and can not be used as a filter')

        return v

    def make_filter(self, document_type: Union[DocumentType, list[DocumentType]]) -> dict:
        filter = {}
        for key, value in self.filters.items():
            if not isinstance(value, Filter):
                raise ValueError(f'Filter {key} is not a valid filter')
            filter[key] = {value.filter_type: value.value}

        if isinstance(document_type, list):
            if DocumentType.ALL not in document_type:
                filter['type'] = {FilterType.IN: document_type}
            else :
                raise ValueError("ALL can not be used with other types")
        else:
            if document_type != DocumentType.ALL:
                filter['type'] = {FilterType.EQUAL: document_type}

        return filter

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

async def get_related_documents(store_name: str, agent: BaseAgent, embedding: Embeddings, nb_results: int, search_filters: SearchFilter, document_type: Union[DocumentType, list[DocumentType]]) -> list[Document]:

    filter = search_filters.make_filter(document_type = document_type)
    related_documents = await agent.vectorstores[store_name].asimilarity_search_by_vector(
        embedding,
        k=nb_results,
        include_metadata=True,
        filter=filter,
    )

    return related_documents

async def search_from_uri(agent : BaseAgent, query: str, uri: str, nb_results: int = 5) -> list[Document]:

    k = nb_results
    if( False ): #TODO: Make an option for clutered search
        k = nb_results * 10
    LOG.notice(f"Clustered search deactivated, k = {k}")

    documents = await  get_related_documents(store_name=DocumentType.DOCUMENTS,
                                        agent=agent,
                                        embedding=await agent.embedding_model.aembed_query(query),
                                        nb_results=k,
                                        search_filters=SearchFilter(filters={'source': Filter(filter_type=FilterType.START_WITH, value=uri)}),
                                        document_type=[DocumentType.DOCUMENTS]
                                        )

    if len(documents) > nb_results and False: #TODO: Make an option for clutered search
        documents =  _get_documents_from_different_cluser(nb_results = nb_results, documents = documents)

    return sorted(documents, key=lambda x: x.metadata['created_at'], reverse=True)

def _get_documents_from_different_cluser(nb_results : int, documents : list[Document]) -> list[Document]:

    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=nb_results, random_state=42).fit([doc.embedding for doc in documents])
    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(nb_results):

        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(documents - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        #closest_indices.append(closest_index)
        closest_indices.append(documents[closest_index])

    return closest_indices


# from langchain.text_splitter import (
#     Language,
#     RecursiveCharacterTextSplitter,
# )
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings

# text_splitter = SemanticChunker(OpenAIEmbeddings())
