"""Tools to interact with the user"""

from __future__ import annotations

# from AFAAS.lib.app import clean_input
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.core.adapters.embeddings.search import get_related_documents, DocumentType, SearchFilter, Filter, FilterType
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.tools.base import AbstractTool
from AFAAS.lib.message_agent_user import Emiter, MessageAgentUser
from AFAAS.lib.message_common import AFAASMessageStack
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema
from langchain_core.documents import Document

@tool(
    name="user_interaction",
    description=(
        "Ask a question to the user if you need more details or information regarding the given goals,"
        " you can ask the user for input"
    ),
    parameters={
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The question or prompt to the user",
            required=True,
        )
    },
    categories=[AbstractTool.FRAMEWORK_CATEGORY],
)
async def user_interaction(
    query: str, task: Task, agent: BaseAgent, skip_proxy=False
) -> str:

    original_query = query
    query_message = MessageAgentUser(
            emitter=Emiter.AGENT.value,
            user_id=agent.user_id,
            agent_id=agent.agent_id,
            message=str(original_query),
        )
    response_message = MessageAgentUser(
            emitter=Emiter.USER.value,
            user_id=agent.user_id,
            agent_id=agent.agent_id,
            message="place_holder",
        )

    proxy_has_answer = False
    if not skip_proxy:
        # NOTE: Design choice: The agent will only use chat messages as source of information & will not use other documents
        query_embedding = await agent.embedding_model.aembed_query(
                text=original_query
            )

        messages = await get_related_documents(store_name ="message_agent_user",
                                         agent=agent,
                                         embedding =  query_embedding ,
                                         nb_results = 10 ,
                                         document_type = DocumentType.ALL,
                                         search_filters= SearchFilter(filters = {
                                            'agent_id' : Filter( 
                                                filter_type=FilterType.EQUAL,
                                                value=agent.agent_id,
                                            ) 
                                         }
                                         )
        )

        if len(messages) > 0 : 
            llm_response =await agent.execute_strategy(strategy_name="user_proxy", 
                                                       documents = messages, 
                                                       query=original_query,
                                                       task=task,
                                                       agent=agent,
                                                       )

            from AFAAS.prompts.common.user_proxy import UserProxyStrategyFunctionNames
            if llm_response.parsed_result[0]['command_name'] != UserProxyStrategyFunctionNames.USER_INTERACTION.value :  # If the user proxy found an answer
                proxy_has_answer = True
                query_message.hidden = True
                response_message.hidden = True
                response_message.message = llm_response.parsed_result[0]['command_args']['answer']
            else : 
                query = llm_response.parsed_result[0]['command_args']['query']



    if proxy_has_answer :
        await agent._user_message_handler(original_query)
        await agent._user_message_handler(response_message.message)
    else :
        user_response = await agent._user_input_handler(query)
        response_message.message = user_response

    await agent.message_agent_user.db_create(
        message=query_message
    )
    await agent.message_agent_user.db_create(
        message=response_message
    )

    import datetime
    import uuid
    document = Document(page_content=f"Question: {original_query}\nAnswer: {response_message.message}")
    document.metadata["type"] = DocumentType.MESSAGE_AGENT_USER.value
    document.metadata["created_at"] = str(datetime.datetime.now())
    document.metadata["document_id"] = str(str(uuid.uuid4()))
    document.metadata["question"] = original_query
    document.metadata["answer"] = response_message.message
    document.metadata["agent_id"] = agent.agent_id
    document.metadata["user_id"] = agent.user_id
    await agent.vectorstores['message_agent_user'].aadd_documents([document])

    return response_message.message
