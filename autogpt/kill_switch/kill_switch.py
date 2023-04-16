import requests
from bs4 import BeautifulSoup


def check_kill_switch(kill_switch_url, kill_switch_canary):
    """Return True if the switch was flicked, this will halt execution of continuous mode. Provide an explanation too."""
    try:
        response = requests.get(kill_switch_url)
        soup = BeautifulSoup(response.content, "html.parser")
        if kill_switch_canary in soup.get_text():
            return False, "Canary found"
        else:
            return True, "Canary missing from kill switch webpage!"
    except Exception as e:
        print(e)
        # Err on the side of safety
        # if we were not able to access the kill switch URL flick the kill switch
        return True, "Could not load canary webpage"
