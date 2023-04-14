import browse
import json
from memory import get_memory
import datetime
import agent_manager as agents
import speak
from config import Config
import ai_functions as ai
from file_operations import read_file, write_to_file, append_to_file, delete_file, search_files
from execute_code import execute_python_file, execute_shell
from json_parser import fix_and_parse_json
from image_gen import generate_image
from duckduckgo_search import ddg
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

cfg = Config()


def is_valid_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def get_command(response):
    """解析响应并返回命令名称和参数"""
    try:
        response_json = fix_and_parse_json(response)

        if "command" not in response_json:
            return "Error:","JSON 中缺少'命令'对象"

        command = response_json["command"]

        if "name" not in command:
            return "Error:", "'command'对象中缺少'name'字段 "

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error:", "无效的 JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error:", str(e)


def execute_command(command_name, arguments):
    """执行命令并返回结果"""
    memory = get_memory(cfg)

    try:
        if command_name == "google":

            # Check if the Google API key is set and use the official search method
            # If the API key is not set or has only whitespaces, use the unofficial search method
            if cfg.google_api_key and (cfg.google_api_key.strip() if cfg.google_api_key else None):
                return google_official_search(arguments["input"])
            else:
                return google_search(arguments["input"])
        elif command_name == "memory_add":
            return memory.add(arguments["string"])
        elif command_name == "start_agent":
            return start_agent(
                arguments["name"],
                arguments["task"],
                arguments["prompt"])
        elif command_name == "message_agent":
            return message_agent(arguments["key"], arguments["message"])
        elif command_name == "list_agents":
            return list_agents()
        elif command_name == "delete_agent":
            return delete_agent(arguments["key"])
        elif command_name == "get_text_summary":
            return get_text_summary(arguments["url"], arguments["question"])
        elif command_name == "get_hyperlinks":
            return get_hyperlinks(arguments["url"])
        elif command_name == "read_file":
            return read_file(arguments["file"])
        elif command_name == "write_to_file":
            return write_to_file(arguments["file"], arguments["text"])
        elif command_name == "append_to_file":
            return append_to_file(arguments["file"], arguments["text"])
        elif command_name == "delete_file":
            return delete_file(arguments["file"])
        elif command_name == "search_files":
            return search_files(arguments["directory"])
        elif command_name == "browse_website":
            return browse_website(arguments["url"], arguments["question"])
        # TODO: Change these to take in a file rather than pasted code, if
        # non-file is given, return instructions "Input should be a python
        # filepath, write your code to file and try again"
        elif command_name == "evaluate_code":
            return ai.evaluate_code(arguments["code"])
        elif command_name == "improve_code":
            return ai.improve_code(arguments["suggestions"], arguments["code"])
        elif command_name == "write_tests":
            return ai.write_tests(arguments["code"], arguments.get("focus"))
        elif command_name == "execute_python_file":  # Add this command
            return execute_python_file(arguments["file"])
        elif command_name == "execute_shell":
            if cfg.execute_local_commands:
                return execute_shell(arguments["command_line"])
            else:
                return "不允许运行本地 shell 命令。 要执行 shell 命令，必须在您的配置中将 EXECUTE_LOCAL_COMMANDS 设置为“True”。 不要试图绕过限制"
        elif command_name == "generate_image":
            return generate_image(arguments["prompt"])
        elif command_name == "do_nothing":
            return "No action performed."
        elif command_name == "task_complete":
            shutdown()
        else:
            return f"未知的命令 '{command_name}'. 请参阅“COMMANDS”列表以获取可用命令,并仅以指定的 JSON 格式响应。"
    # All errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)


def get_datetime():
    """返回当前日期和时间"""
    return "当前日期和时间: " + \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def google_search(query, num_results=8):
    """Return谷歌搜索的结果"""
    search_results = []
    for j in ddg(query, max_results=num_results):
        search_results.append(j)

    return json.dumps(search_results, ensure_ascii=False, indent=4)


def google_official_search(query, num_results=8):
    """Return 使用官方 Google API 的谷歌搜索结果"""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import json

    try:
        # Get the Google API key and Custom Search Engine ID from the config file
        api_key = cfg.google_api_key
        custom_search_engine_id = cfg.custom_search_engine_id

        # Initialize the Custom Search API service
        service = build("customsearch", "v1", developerKey=api_key)

        # Send the search query and retrieve the results
        result = service.cse().list(q=query, cx=custom_search_engine_id, num=num_results).execute()

        # Extract the search result items from the response
        search_results = result.get("items", [])

        # Create a list of only the URLs from the search results
        search_results_links = [item["link"] for item in search_results]

    except HttpError as e:
        # Handle errors in the API call
        error_details = json.loads(e.content.decode())

        # Check if the error is related to an invalid or missing API key
        if error_details.get("error", {}).get("code") == 403 and "invalid API key" in error_details.get("error", {}).get("message", ""):
            return "Error: 提供的 Google API Key无效或丢失。"
        else:
            return f"Error: {e}"

    # Return the list of search result URLs
    return search_results_links


def browse_website(url, question):
    """浏览网站并返回摘要和链接"""
    summary = get_text_summary(url, question)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    result = f"""网站内容概要: {summary}\n\nLinks: {links}"""

    return result


def get_text_summary(url, question):
    """谷歌搜索的结果"""
    text = browse.scrape_text(url)
    summary = browse.summarize_text(text, question)
    return """ "Result" : """ + summary


def get_hyperlinks(url):
    """返回谷歌搜索的结果"""
    link_list = browse.scrape_links(url)
    return link_list


def commit_memory(string):
    """将字符串提交到内存"""
    _text = f"""用字符串提交内存 "{string}" """
    mem.permanent_memory.append(string)
    return _text


def delete_memory(key):
    """使用指定的key删除内存"""
    if key >= 0 and key < len(mem.permanent_memory):
        _text = "Deleting memory with key " + str(key)
        del mem.permanent_memory[key]
        print(_text)
        return _text
    else:
        print("无效Key，无法删除内存。")
        return None


def overwrite_memory(key, string):
    """用指定的键和字符串覆盖内存"""
    # Check if the key is a valid integer
    if is_valid_int(key):
        key_int = int(key)
        # Check if the integer key is within the range of the permanent_memory list
        if 0 <= key_int < len(mem.permanent_memory):
            _text = "覆盖 memory with key " + str(key) + " and string " + string
            # Overwrite the memory slot with the given integer key and string
            mem.permanent_memory[key_int] = string
            print(_text)
            return _text
        else:
            print(f"无效的Key '{key}', out of range.")
            return None
    # Check if the key is a valid string
    elif isinstance(key, str):
        _text = "覆盖 memory with key " + key + " and string " + string
        # Overwrite the memory slot with the given string key and string
        mem.permanent_memory[key] = string
        print(_text)
        return _text
    else:
        print(f"无效的Key '{key}', 必须是整数或字符串.")
        return None


def shutdown():
    """关闭程序"""
    print("关闭...")
    quit()


def start_agent(name, task, prompt, model=cfg.fast_llm_model):
    """使用name、task和prompt来启动机器人"""
    global cfg

    # Remove underscores from name
    voice_name = name.replace("_", " ")

    first_message = f"""你是 {name}.  回应: "Acknowledged". 中文版来自自阿杰，公众号内获取最新代码《阿杰的人生路》"""
    agent_intro = f"{voice_name} here, Reporting for duty!"

    # Create agent
    if cfg.speak_mode:
        speak.say_text(agent_intro, 1)
    key, ack = agents.create_agent(task, first_message, model)

    if cfg.speak_mode:
        speak.say_text(f"Hello {voice_name}. 你的任务如下： {task}.")

    # Assign task (prompt), get response
    agent_response = message_agent(key, prompt)

    return f"Agent {name} created with key {key}. First response: {agent_response}"


def message_agent(key, message):
    """使用指定的密钥和消息向机器人发送消息"""
    global cfg

    # Check if the key is a valid integer
    if is_valid_int(key):
        agent_response = agents.message_agent(int(key), message)
    # Check if the key is a valid string
    elif isinstance(key, str):
        agent_response = agents.message_agent(key, message)
    else:
        return "无效的Key, 必须是整数或字符串."

    # Speak response
    if cfg.speak_mode:
        speak.say_text(agent_response, 1)
    return agent_response


def list_agents():
    """列出所有机器人"""
    return agents.list_agents()


def delete_agent(key):
    """删除具有指定Key的机器人"""
    result = agents.delete_agent(key)
    if not result:
        return f"机器人 {key} 不存在."
    return f"机器人 {key} 已经删除啦，放心了吧."
