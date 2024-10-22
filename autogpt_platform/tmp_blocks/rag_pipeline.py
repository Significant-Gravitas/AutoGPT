import logging
from enum import Enum
from json import JSONDecodeError
from typing import Any, List, NamedTuple

import anthropic
import ollama
import openai
import requests
from groq import Groq

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField
from backend.util import json
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class RAGTechnique(str, Enum):
    BASIC = "basic",
    COT = "chain_of_thought"
    HYDE = "hypothetical_document"
    MULTI_QUERY = "multi_query"

class RagPipelineBlock(Block):
    class Input(BlockSchema):
        index_name: str = SchemaField(description="Name of the Pinecone Index")
        pinecone_api_key: BlockSecret = SecretField(key="pinecone_api_key", description="Pinecone API Key")
        jina_api_key: BlockSecret = SecretField(key="jina_api_key", description="Jina API Key")
        openai_api_key: BlockSecret = SecretField(key="openai_api_key", description="OpenAI API Key")
        query: str = SchemaField(description="Natural language query about a topic")
        namespace: str = SchemaField(description="Namespace of the topic")
        top_k: str = SchemaField(description="Number of top results to return")
        rag_technique: str = SchemaField(description="RAG technique to use", default=RAGTechnique.BASIC)

    class Output(BlockSchema):
        response: dict[str, Any]
        error: str

    def __init__(self):
        super().__init__(
            id="0cfcc32b-4526-4729-adb1-2a4628d66feb",
            description="Block to query data from pinecone",
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            input_schema=RagPipelineBlock.Input,
            output_schema=RagPipelineBlock.Output,
        )

    def get_embeddings(self, text: str, api_key: str) -> list:
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = {
            'input': [text],
            'model': 'jina-embeddings-v3'
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

    def query_pinecone(self, index_name: str, api_key: str, vector: list, namespace: str, top_k: int) -> list:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        results = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
        return results.matches

    def generate_hypothetical_document(self, query: str, api_key:str) -> str:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": "You are an AI that generates hypothetical documents based on queries from the user."},
                {"role": "user", "content": f"write a passage containing information about the following query: {query}"}
            ],
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7
        )
    def hyde_technique(self, query: str, api_keys: dict) -> str:
        hyde_document = self.generate_hypothetical_document(query, api_keys['openai_api_key'])
        hyde_embedding = self.get_embeddings(hyde_document, api_keys['jina'])
        results = self.query_pinecone(
            self.input_data.index_name,
            api_keys['pinecone_api_key'],
            hyde_embedding,
            self.input_data.namespace,
            self.input_data.top_k,
        )
        context = "\n".join(results['metadata']['text'] for result in results)
        prompt = f"Based on the following information and only the following information, please answer the question: '{query}'\n\nContext:\n{context}\n\nAnswer:"
        openai.api_key = api_keys['openai']
        response = openai.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user",
                 "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()


    def run(self, input_data: Input) -> BlockOutput:
        self.input_data = input_data
        api_keys = {
            'openai': input_data.openai_api_key.get_secret_value(),
            'pinecone': input_data.pinecone_api_key.get_secret_value(),
            'jina': input_data.jina_api_key.get_secret_value(),
        }
        try:
            if input_data.rag_technique == RAGTechnique.HYDE:
                response = self.hyde_technique(input_data.query, api_keys)
            else:
                raise ValueError(f"Unknown RAG technique {input_data.rag_technique}")
            yield "response", response
            yield "technique_used", input_data.rag_technique
        except Exception as e:
            error_message = f"error during query: {str(e)}"
            yield "error", error_message
            yield "I'm sorry something went wrong when trying to answer your query"
            yield "technique_used", "none"



