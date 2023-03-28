import browse
import json
import memory as mem

### Implemented Commands: ###

def google_search(query, num_results = 3):
    search_results = []
    for j in browse.search(query, num_results=num_results):
        search_results.append(j)
    
    return json.dumps(search_results, ensure_ascii=False, indent=4)

def transcribe_summarise(url):
    text = mem.scrape_main_content(url)
    summary = mem.summarize_text(text)
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