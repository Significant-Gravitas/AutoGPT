import browse
import json
import memory as mem
import datetime

def get_command(response):
    try:
        response_json = json.loads(response)
        command = response_json["command"]
        command_name = command["name"]
        arguments = command["args"]

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
        elif command_name == "start_instance":
            return start_instance(arguments["name"], arguments["prompt"])
        elif command_name == "manage_instances":
            return manage_instances(arguments["action"])
        elif command_name == "navigate_website":
            return navigate_website(arguments["action"], arguments["username"])
        elif command_name == "register_account":
            return register_account(arguments["username"], arguments["website"])
        elif command_name == "transcribe_summarise":
            return transcribe_summarise(arguments["url"])
        else:
            return f"unknown command {command_name}"
    # All other errors, return "Error: + error message"
    except Exception as e:
        return "Error: " + str(e)

def get_datetime():
    return "Current date and time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


### Implemented Commands: ###
def google_search(query, num_results = 3):
    search_results = []
    for j in browse.search(query, num_results=num_results):
        search_results.append(j)
    
    return json.dumps(search_results, ensure_ascii=False, indent=4)

def transcribe_summarise(url):
    text = browse.scrape_text(url)
    summary = browse.summarize_text(text)
    return """ "Result" : """ + summary

def check_news(source):
    print("Checking news from BBC world instead of " + source)
    _text= transcribe_summarise("https://www.bbc.com/news/world")
    return _text

def commit_memory(string):
    _text = "Committing memory with string " + string
    mem.permanent_memory.append(string)
    print(_text)
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
    

### TODO: Not Yet Implemented: ###

def start_instance(name, prompt):
    _text = "Starting instance with name " + name + " and prompt " + prompt
    print(_text)
    return "Command not implemented yet."

def manage_instances(action):
    _text = "Managing instances with action " + action
    print(_text)
    return _text

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