import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.vectorstores import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import ValidationError

from AFAAS.interfaces.adapters.embeddings.wrapper import (
    ChromaWrapper,
    Document,
    DocumentType,
    Filter,
    FilterType,
    SearchFilter,
    VectorStoreWrapper,
)
from tests.dataset.plan_familly_dinner import (
    Task,
    _plan_familly_dinner,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
    task_ready_no_predecessors_or_subtasks,
)

from .dataset.documents import documents

# @pytest.mark.asyncio
# async def test_dataset(documents):
#     i = 0
#     for doc in documents:
#         print(doc)
#         i += 1
#         if i > 10:
#             break


@pytest.fixture
async def vector_store_wrapper(default_task: Task):
    return ChromaWrapper(
        vector_store=Chroma(
            persist_directory=f"data/chroma/", embedding_function=OpenAIEmbeddings()
        ),
        embedding_model=OpenAIEmbeddings(),
    )


@pytest.fixture
async def vector_store_wrapper_with_documents(documents: list[Document]):
    wrapper = ChromaWrapper(
        vector_store=Chroma(
            persist_directory=f"data/chroma/regerg",  # + str(uuid.uuid4()),
            embedding_function=OpenAIEmbeddings(),
        ),
        embedding_model=OpenAIEmbeddings(),
    )
    for doc in documents:
        doc.metadata = {"agent_id": "123"}
        await wrapper.add_document(
            document_type=DocumentType.DOCUMENTS,
            document=doc,
            document_id="DOC" + str(uuid.uuid4()),
        )
        print(doc)
    return wrapper


@pytest.mark.asyncio
async def test_add_document_valid_input(vector_store_wrapper: VectorStoreWrapper):
    # Mock async call to vector_store.aadd_documents
    vector_store_wrapper.vector_store.aadd_documents = AsyncMock(
        return_value=["doc_id_1"]
    )

    document = Document(page_content="Sample content", metadata={"agent_id": "123"})
    doc_id = await vector_store_wrapper.add_document(
        document_type=DocumentType.DOCUMENTS, document=document, document_id="doc_id_1"
    )

    assert doc_id == "doc_id_1", "Document ID should be returned correctly"


@pytest.mark.asyncio
async def test_add_document_missing_metadata(vector_store_wrapper: VectorStoreWrapper):
    document = Document(page_content="Sample content", metadata={})
    with pytest.raises(ValueError) as excinfo:
        await vector_store_wrapper.add_document(
            document_type=DocumentType.DOCUMENTS, document=document
        )

    assert (
        "At least one of 'plan_id', 'agent_id', or 'user_id' must be provided"
        in str(excinfo.value)
    )


@pytest.mark.asyncio
async def test_get_related_documents_test_query_no_result(
    vector_store_wrapper_with_documents: VectorStoreWrapper,
):
    search_filter = SearchFilter(
        filters={"agent_id": Filter(filter_type=FilterType.EQUAL, value="123")}
    )
    results = await vector_store_wrapper_with_documents.get_related_documents(
        query="test query",  # await vector_store_wrapper_with_documents.embedding_model.aembed_query("test query"),
        nb_results=5,
        similarity_threshold=0.99999,
        search_filters=search_filter,
        document_type=DocumentType.DOCUMENTS,
    )

    assert isinstance(results, list), "Results should be a list"
    assert len(results) == 0, "Results list should be empty"
    # Additional checks can be added to verify the contents of the results


@pytest.mark.asyncio
async def test_get_related_documents_related_query(
    vector_store_wrapper_with_documents: VectorStoreWrapper,
):
    # async def test_get_related_documents_related_query(documents : list[Document]):
    #     wrapper =  ChromaWrapper(vector_store=Chroma(
    #                 persist_directory=f'data/chroma/regerg', #+ str(uuid.uuid4()),
    #                 embedding_function=OpenAIEmbeddings()
    #             ), embedding_model= OpenAIEmbeddings()
    #             )
    #     for doc in documents:
    #         doc.metadata = {"agent_id": "123"}
    #         doc_id = await wrapper.add_document(document_type= DocumentType.DOCUMENTS , document=  doc)
    search_filter = SearchFilter(
        filters={"agent_id": Filter(filter_type=FilterType.EQUAL, value="123")}
    )
    results = await vector_store_wrapper_with_documents.get_related_documents(
        query="god creation",  # await vector_store_wrapper_with_documents.embedding_model.aembed_query("god creation"),
        nb_results=20,
        similarity_threshold=0.01,
        search_filters=search_filter,
        document_type=DocumentType.DOCUMENTS,
    )

    assert isinstance(results, list), "Results should be a list"
    assert len(results) == 20, "Results list 20 items long"
    # Additional checks can be added to verify the contents of the results


@pytest.mark.asyncio
async def test_get_related_documents_no_matching_results(
    vector_store_wrapper_with_documents: VectorStoreWrapper,
):
    search_filter = SearchFilter(
        filters={"agent_id": Filter(filter_type=FilterType.EQUAL, value="nonexistent")}
    )
    results = await vector_store_wrapper_with_documents.get_related_documents(
        query="test query",
        nb_results=5,
        search_filters=search_filter,
        document_type=DocumentType.DOCUMENTS,
    )

    assert isinstance(results, list), "Results should be a list"
    assert len(results) == 0, "Results list should be empty for non-matching criteria"


def test_get_related_documents_invalid_filter():
    with pytest.raises(ValidationError) as excinfo:
        SearchFilter(
            filters={"invalid_filter": Filter(filter_type="invalid_type", value="123")}
        )


@pytest.mark.asyncio
async def test_get_related_documents_cluster_search(
    vector_store_wrapper_with_documents: VectorStoreWrapper,
):
    search_filter = SearchFilter(
        filters={"agent_id": Filter(filter_type=FilterType.EQUAL, value="123")}
    )
    results = await vector_store_wrapper_with_documents.get_related_documents(
        query="test query",
        nb_results=5,
        search_filters=search_filter,
        document_type=DocumentType.DOCUMENTS,
        cluster_search=True,
    )

    assert isinstance(results, list), "Results should be a list"
    # Check if results are clustered appropriately
