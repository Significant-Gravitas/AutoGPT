import openai
import revChatGPT
from revChatGPT.V1 import Chatbot
from config import Config
cfg = Config()

openai.api_key = cfg.openai_api_key
chatbot = Chatbot(config={
  "access_token": cfg.access_token,
  "paid": True,
  "lazy_loading": False,
})
print("Chatbot: ")
# Overly simple abstraction until we create something better
def create_chat_completion_open_api(model=None, messages=None, max_tokens=None, temperature=None)->str:
    """Create a chat completion using the OpenAI API"""
    if cfg.use_azure:
        response = openai.ChatCompletion.create(
            deployment_id=cfg.openai_deployment_id,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return response.choices[0].message["content"]

def create_chat_completion(model=None, messages=None, max_tokens=None, temperature=None):
    response = ""
    # logging.info(f"conversation_mapping-{chatbot.conversation_mapping}, conversation_id - {chatbot.conversation_id}, parent_id - {chatbot.parent_id}")
    # TODO：如果出现 gpt4 限流的问题，需要降级到 gpt-3.5-turbo
    # 从 messages 数组中提取 content 内容
    message_contents = " ".join([msg["content"] for msg in messages])
    print(f"message_contents =>>> {message_contents}")
    try:
        for data in chatbot.ask(
            prompt=message_contents,
            conversation_id=chatbot.conversation_id,
            parent_id=chatbot.parent_id,
            model="gpt-4"
        ):
            response = data["message"]
        print(f"get_answer_from_chatGPT => {response}")
    except revChatGPT.typings.Error as e:
        # 429错误处理
        if "model_cap_exceeded" in str(e):
            print("Too Many Requests. model_cap_exceeded...")
            # 降级到较低版本的API
            for data in chatbot.ask(
                prompt=message_contents,
                conversation_id=chatbot.conversation_id,
                parent_id=chatbot.parent_id,
                model="gpt-3.5-turbo"
            ):
                response = data["message"]
            print(f"get_answer_from_chatGPT (fallback to gpt-3.5-turbo) => {response}")
        else:
            raise e
    # 把 parent_thread_id 和 chatbot.parent_id 对应关系写入文件系统中，下次进来直接从文件系统根据 parent_thread_id 读取这个 parent_id
    return response
