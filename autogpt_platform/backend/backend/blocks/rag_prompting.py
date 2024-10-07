from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField, BlockSecret, SecretField
from pinecone import Pinecone
import requests
import openai
import uuid
from enum import Enum
import ollama


class RAGTechnique(str, Enum):
    BASIC = "basic"
    COT = "chain_of_thought"
    HYDE = "hypothetical_document"
    MULTI_QUERY = "multi_query"


class RAGPromptingBlock(Block):
    class Input(BlockSchema):
        index_name: str = SchemaField(description="Name of the Pinecone index")
        pinecone_api_key: BlockSecret = SecretField(key="pinecone_api_key", description="Pinecone API Key")
        jina_api_key: BlockSecret = SecretField(key="jina_api_key", description="Jina API Key")
        openai_api_key: BlockSecret = SecretField(key="openai_api_key", description="OpenAI API Key")
        query: str = SchemaField(description="Natural language query")
        namespace: str = SchemaField(description="Namespace to query in Pinecone", default="")
        top_k: int = SchemaField(description="Number of top results to retrieve", default=3)
        rag_technique: RAGTechnique = SchemaField(description="RAG technique to use", default=RAGTechnique.BASIC)

    class Output(BlockSchema):
        response: str = SchemaField(description="Natural language response based on retrieved information")
        technique_used: str = SchemaField(description="RAG technique used for this query")
        error: str = SchemaField(description="Error message if query fails", default="")

    def __init__(self):
        super().__init__(
            id="e9aeec7e-6333-44e7-a80e-4846f2a0b60b",
            description="Advanced Pinecone query block with multiple RAG techniques",
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            input_schema=RAGPromptingBlock.Input,
            output_schema=RAGPromptingBlock.Output,
        )

    def get_embedding(self, text: str, api_key: str) -> list:
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = {
            'input': [text],
            'model': 'jina-embeddings-v2-base-en'
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def query_pinecone(self, index_name: str, api_key: str, vector: list, namespace: str, top_k: int) -> list:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        results = index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
        return results.matches

    def generate_hypothetical_document(self, query: str, api_key: str) -> str:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI that generates hypothetical documents based on queries."},
                {"role": "user",
                 "content": f"Write a passage containing information about the following query: {query}"}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def generate_sub_queries(self, query: str, api_key: str, num_queries: int = 3) -> list:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an AI that generates similar sub-queries based on an original query."},
                {"role": "user",
                 "content": f"Generate {num_queries} similar sub-queries for the following query: {query}"}
            ],
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )
        sub_queries = response.choices[0].message.content.strip().split("\n")
        return [sq.split(". ", 1)[-1] for sq in sub_queries]  # Remove numbering if present

    def basic_technique(self, query: str, api_keys: dict) -> str:
        query_embedding = self.get_embedding(query, api_keys['jina'])
        results = self.query_pinecone(
            self.input_data.index_name,
            api_keys['pinecone'],
            query_embedding,
            self.input_data.namespace,
            self.input_data.top_k
        )
        context = "\n".join([result['metadata']['text'] for result in results])
        prompt = f"Based on the following information, please answer the question: '{query}'\n\nContext:\n{context}\n\nAnswer:"

        openai.api_key = api_keys['openai']
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7,
        )
        # response = ollama.chat(
        #     model="qwen2.5",
        #     messages=[
        #         {"role": "system",
        #          "content": "You are a helpful assistant that answers questions based on the provided context."},
        #         {"role": "user", "content": prompt}
        #     ],
        # )
        return response.choices[0].message.content.strip()

    def chain_of_thought_technique(self, query: str, api_keys: dict) -> str:
        # Retrieve relevant information
        query_embedding = self.get_embedding(query, api_keys['jina'])
        results = self.query_pinecone(
            self.input_data.index_name,
            api_keys['pinecone'],
            query_embedding,
            self.input_data.namespace,
            self.input_data.top_k
        )
        context = "\n".join([result['metadata']['text'] for result in results])

        # Construct the CoT prompt
        cot_prompt = f"""To answer the question: '{query}', let's approach this step-by-step using the following information:

Context:
{context}

Please follow these steps:
1. Identify the key elements of the question.
2. Analyze the relevant information from the context.
3. Form a logical chain of reasoning.
4. Arrive at a conclusion.

Provide your thought process for each step, then give the final answer.

Step-by-step reasoning:"""

        openai.api_key = api_keys['openai']
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that uses chain-of-thought reasoning to answer questions based on provided context."},
                {"role": "user", "content": cot_prompt}
            ],
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def hyde_technique(self, query: str, api_keys: dict) -> str:
        hyde_doc = self.generate_hypothetical_document(query, api_keys['openai'])
        hyde_embedding = self.get_embedding(hyde_doc, api_keys['jina'])
        results = self.query_pinecone(
            self.input_data.index_name,
            api_keys['pinecone'],
            hyde_embedding,
            self.input_data.namespace,
            self.input_data.top_k
        )
        context = "\n".join([result['metadata']['text'] for result in results])
        prompt = f"Based on the following information, please answer the question: '{query}'\n\nContext:\n{context}\n\nAnswer:"

        openai.api_key = api_keys['openai']
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def multi_query_technique(self, query: str, api_keys: dict) -> str:
        # Generate sub-queries
        sub_queries = self.generate_sub_queries(query, api_keys['openai'])

        # Retrieve information for each sub-query and the original query
        all_contexts = []
        for q in [query] + sub_queries:
            embedding = self.get_embedding(q, api_keys['jina'])
            results = self.query_pinecone(
                self.input_data.index_name,
                api_keys['pinecone'],
                embedding,
                self.input_data.namespace,
                self.input_data.top_k
            )
            context = "\n".join([result['metadata']['text'] for result in results])
            all_contexts.append(f"Query: {q}\nContext: {context}\n")

        # Combine all contexts
        combined_context = "\n".join(all_contexts)

        # Generate final answer using all retrieved information
        prompt = f"""Based on the following information from multiple related queries, please provide a comprehensive answer to the original question: '{query}'

Context from multiple queries:
{combined_context}

Comprehensive Answer:"""

        openai.api_key = api_keys['openai']
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that provides comprehensive answers based on information from multiple related queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def run(self, input_data: Input) -> BlockOutput:
        self.input_data = input_data
        api_keys = {
            'openai': input_data.openai_api_key.get_secret_value(),
            'pinecone': input_data.pinecone_api_key.get_secret_value(),
            'jina': input_data.jina_api_key.get_secret_value()
        }

        try:
            if input_data.rag_technique == RAGTechnique.BASIC:
                response = self.basic_technique(input_data.query, api_keys)
            elif input_data.rag_technique == RAGTechnique.HYDE:
                response = self.hyde_technique(input_data.query, api_keys)
            elif input_data.rag_technique == RAGTechnique.MULTI_QUERY:
                response = self.multi_query_technique(input_data.query, api_keys)
            elif input_data.rag_technique == RAGTechnique.COT:
                response = self.chain_of_thought_technique(input_data.query, api_keys)
            else:
                raise ValueError(f"Unknown RAG technique: {input_data.rag_technique}")

            yield "response", response
            yield "technique_used", input_data.rag_technique.value
        except Exception as e:
            error_message = f"Error during query process: {str(e)}"
            yield "error", error_message
            yield "response", "I'm sorry, but I encountered an error while processing your query."
            yield "technique_used", "none"