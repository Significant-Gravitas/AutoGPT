import hashlib
import shutil

import pytest

from forge.memory.chroma_memstore import ChromaMemStore


@pytest.fixture
def memstore():
    mem = ChromaMemStore(".test_mem_store")
    yield mem
    shutil.rmtree(".test_mem_store")


def test_add(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    assert memstore.client.get_or_create_collection(task_id).count() == 1


def test_query(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    query = "test"
    assert len(memstore.query(task_id, query)["documents"]) == 1


def test_update(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    updated_document = "This is an updated test document."
    updated_metadatas = {"metadata": "updated_test_metadata"}
    memstore.update(task_id, [doc_id], [updated_document], [updated_metadatas])
    assert memstore.get(task_id, [doc_id]) == {
        "documents": [updated_document],
        "metadatas": [updated_metadatas],
        "embeddings": None,
        "ids": [doc_id],
    }


def test_delete(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    memstore.delete(task_id, doc_id)
    assert memstore.client.get_or_create_collection(task_id).count() == 0
