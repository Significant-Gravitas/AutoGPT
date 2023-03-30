import browse
import json
import memory as mem
import datetime
import agent_manager as agents


def get_command(response):
    try:
        response_json = json.loads(response)
        command = response_json["command"]
        command_name = command["name"]
        arguments = command["args"]

        if not arguments:
            arguments = {}

        return command_name, arguments
    except json.decoder.JSONDecodeError:
        return "Error: Invalid JSON"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)

def execute_command(command_name, arguments):
    try:
        if command_name == "google":
            return google_search(arguments["input"])
        elif command_name == "check_news":
            return check_news(arguments["source"])
        elif command_name == "check_notifications":
            return check_notifications(arguments["website"])
        elif command_name == "memory_add":
            return commit_memory(arguments["string"])
        elif command_name == "memory_del":
            return delete_memory(arguments["key"])
        elif command_name == "memory_ovr":
            return overwrite_memory(arguments["key"], arguments["string"])
        elif command_name == "start_agent":
            return start_agent(arguments["task"], arguments["prompt"])
        elif command_name == "message_agent":
            return message_agent(arguments["key"], arguments["message"])
        elif command_name == "list_agents":
            return list_agents()
        elif command_name == "delete_agent":
            return delete_agent(arguments["key"])
        elif command_name == "navigate_website":
            return navigate_website(arguments["action"], arguments["username"])
        elif command_name == "register_account":
            return register_account(arguments["username"], arguments["website"])
        elif command_name == "get_text_summary":
            return get_text_summary(arguments["url"])
        elif command_name == "get_hyperlinks":
            return get_hyperlinks(arguments["url"])
        elif command_name == "write_to_file":
            return write_to_file(arguments["file"], arguments["content"])
        elif browse_website == "browse_website":
            return browse_website(arguments["url"])
        elif command_name == "task_complete":
            shutdown()
        else:
            return f"unknown command {command_name}"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)

def get_datetime():
    return "Current date and time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


### Implemented Commands: ###
def google_search(query, num_results = 8):
    search_results = []
    for j in browse.search(query, num_results=num_results):
        search_results.append(j)
    
    return json.dumps(search_results, ensure_ascii=False, indent=4)

def browse_website(url):
    summary = get_text_summary(url)
    links = get_hyperlinks(url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    result = f"""Website Content Summary: {summary}\n\nLinks: {links}"""

    return result

def get_text_summary(url):
    text = browse.scrape_text(url)
    summary = browse.summarize_text(text)
    return """ "Result" : """ + summary

def get_hyperlinks(url):
    link_list = browse.scrape_links(url)
    return link_list

def check_news(source):
    print("Checking news from BBC world instead of " + source)
    _text= get_text_summary("https://www.bbc.com/news/world")
    return _text

def commit_memory(string):
    _text = f"""Committing memory with string "{string}" """
    mem.permanent_memory.append(string)
    return _text

def delete_memory(key):
    if key >= 0 and key < len(mem.permanent_memory):
        _text = "Deleting memory with key " + str(key)
        del mem.permanent_memory[key]
        print(_text)
        return _text
    else:
        print("Invalid key, cannot delete memory.")
        return None
    
def overwrite_memory(key, string):
    if key >= 0 and key < len(mem.permanent_memory):
        _text = "Overwriting memory with key " + str(key) + " and string " + string
        mem.permanent_memory[key] = string
        print(_text)
        return _text
    else:
        print("Invalid key, cannot overwrite memory.")
        return None

def write_to_file(filename, content):
    try:
        f = open(filename, "w")
        f.write(content)
        f.close()
    except Exception as e:
        return "Error: " + str(e)
    return "File written to successfully."

def shutdown():
    print("Shutting down...")
    quit()


    

### TODO: Not Yet Implemented: ###

def start_agent(task, prompt, model = "gpt-3.5-turbo"):
    key, agent_response = agents.create_agent(task, prompt, model)
    return f"Agent created with key {key}. First response: {agent_response}"

def message_agent(key, message):
    agent_response = agents.message_agent(key, message)
    return f"Agent {key} responded: {agent_response}"

def list_agents():
    return agents.list_agents()

def delete_agent(key):
    result = agents.delete_agent(key)
    if result == False:
        return f"Agent {key} does not exist."
    return f"Agent {key} deleted."


def navigate_website(action, username):
    _text = "Navigating website with action " + action + " and username " + username
    print(_text)
    return "Command not implemented yet."

def register_account(username, website):
    _text = "Registering account with username " + username + " and website " + website
    print(_text)
    return "Command not implemented yet."

def check_notifications(website):
    _text = "Checking notifications from " + website
    print(_text)
    return "Command not implemented yet."