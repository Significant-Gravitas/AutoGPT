import pytest
from unittest.mock import AsyncMock, patch
from AFAAS.core.tools.builtins.user_interaction import user_interaction
from AFAAS.interfaces.agent.main import BaseAgent
from tests.dataset.plan_familly_dinner import (
    Task,
    _plan_familly_dinner,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
)
from langchain_core.documents import Document

# Function to create a list of mock documents
def create_mock_documents(num_documents):
    documents = []
    for i in range(num_documents):
        doc = Document(
            page_content=f"This is document {i}.",
            metadata={
                "created_at": str(i),
                "author": f"Author {i}",
                "document_id": str(1000 + i)
            }
        )
        documents.append(doc)
    return documents


# Test with specific scenarios and parameter variations
@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected_response", [
    ("Your test query with proxy answer", "expected response"),
    ("Your test query without proxy answer", "expected user response")
])
async def test_user_interaction_scenarios(query, expected_response, default_task):
    pytest.skip("Not implemented")
    mock_task = default_task
    mock_agent = default_task.agent

    mock_agent.embedding_model
    with patch.object(mock_agent._embedding_model, 'aembed_query', AsyncMock(return_value="mock_embedding")):
        with patch.object(mock_agent, 'execute_strategy', AsyncMock(return_value=create_mock_documents(5 if "proxy answer" in query else 0))):
            response = await user_interaction(query, mock_task, mock_agent)

    # Assertions
    assert response == expected_response
