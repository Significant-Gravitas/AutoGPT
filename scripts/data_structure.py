import os
from typing import List, Union
import pinecone
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


load_dotenv()


def get_text_loaders(directory='scripts/database'):
    """this function searches a directory for files and returns a list of TextLoader objects"""
    loaders = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            loaders.append(TextLoader(filepath, encoding='utf-8'))
    return loaders


embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
INDEX_NAME = "auto-gpt"
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="us-central1-gcp")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docsearch = None

loaders = get_text_loaders()

# Iterate over each TextLoader object
for loader in loaders:
    documents = loader.load()
    document_chunks = text_splitter.split_documents(documents)
    
    # Create or update Pinecone index
    if docsearch is None:
        docsearch = Pinecone.from_documents(document_chunks, embeddings, index_name=INDEX_NAME, upsert=True)
    else:
        docsearch.add_documents(document_chunks, upsert=True)


def get_best_matching_document_content(question):
    """This function returns the best matching document content based on the user's query."""   
    matching_docs = docsearch.similarity_search(question)

    if matching_docs:
        best_document = matching_docs[0]
        return best_document.page_content
    else:
        return None
