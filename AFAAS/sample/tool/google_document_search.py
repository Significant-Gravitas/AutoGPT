from llama_index import VectorStoreIndex, download_loader

from AFAAS.core.tools.tool_decorator import tool

# Assuming 'index' is your VectorStoreIndex instance and is accessible here


@tool(
    name="googledocument_search",
    description="Search an information within a GoogleDocsdocument.",
    parameters={
        "query": {
            "type": "string",
            "description": "The search query string.",
            "required": True,
        },
        "document_id": {
            "type": "string",
            "description": "The ID of the Google Docs document to be searched.",
            "required": True,
        },
        "k": {
            "type": "integer",
            "description": "The number of search results to return. Optional.",
        },
    },
)
def document_search(query: str, document_id: str, k: int = 10):
    GoogleDocsReader = download_loader("GoogleDocsReader")

    gdoc_ids = [document_id]
    loader = GoogleDocsReader()
    documents = loader.load_data(document_ids=gdoc_ids)
    index = VectorStoreIndex.from_documents(documents)

    return index.query(query, k=k)
