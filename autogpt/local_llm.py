
import json
import requests
from autogpt.config.config import Config


class OoobaboogaAPI: 
    """This class implements the singleton pattern in Python."""  
    _instance = None  
    def __new__(cls, *args, **kwargs):  
        if cls._instance is None or cls._instance.__class__!= cls:  
            cls._instance = super(OoobaboogaAPI, cls).__new__(cls)  
        return cls._instance

    def __init__(self, CFG: Config):
        self.host = CFG.oobabooga_api_host
        

    def generate_text(self, input: str, **manual_params) -> None:
        params = {
            'max_new_tokens': 200,
            'do_sample': True,
            'temperature': 0.72,
            'top_p': 0.73,
            'typical_p': 1,
            'repetition_penalty': 1.1,
            'encoder_repetition_penalty': 1.0,
            'top_k': 0,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': [],
        } | manual_params 
        
        payload = json.dumps([input, params])
        post_payload = {
            "data": [
                payload
            ]
        }

        print(f"POST PAYLOAD: {post_payload}")

        response = requests.post(f"{self.host}/run/textgen", json=post_payload).json()
        print(f"RESPONSE: {response}")

        reply = response["data"][0]
        print(f"REPLY: {reply}")
        return reply

