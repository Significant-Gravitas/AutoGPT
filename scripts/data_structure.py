"""this module structures and splits the data then embeds it into Pinecone"""
import os
import pinecone
import yaml
from dotenv import load_dotenv
# from config import Config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# cfg = Config()

load_dotenv()

def get_text_loader(filename, directory='scripts/static'):
    """this fuction takes filename and searches a directory for file then loads with TextLoader"""
    filepath = os.path.join(directory,filename)
    if os.path.isfile(filepath):
        return TextLoader(filepath, encoding='utf-8')
    else:
        return None

# Utilize .yaml settings to find directory of database
SETTINGS_FILE = "ai_settings.yaml"

with open(SETTINGS_FILE, encoding='utf-8') as file:
    config_params = yaml.load(file, Loader=yaml.FullLoader)
 
knowledgebase = config_params.get("knowledgebase", "")
# Instance of TextLoader with the knowledgebase from yaml
loader = get_text_loader(knowledgebase)
# Produces a string to be split
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
document_chunks = text_splitter.split_documents(documents)

#Can utilize config file better
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
INDEX_NAME = "auto-gpt"

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="us-central1-gcp")

docsearch = Pinecone.from_documents(document_chunks, embeddings, index_name=INDEX_NAME)

def get_best_matching_document_content(question):
    """This function returns the best matching document content based on the user's query."""   
    matching_docs = docsearch.similarity_search(question)

    if matching_docs:
        best_document = matching_docs[0]
        return best_document.page_content
    else:
        return None
