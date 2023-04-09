import argparse
from config import Config
from console_interaction import print_to_console
from colorama import Fore, Style


cfg = Config()

def parse_arguments():
    global cfg
    cfg.set_continuous_mode(False)
    cfg.set_speak_mode(False)
    
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--continuous', action='store_true', help='Enable Continuous Mode')
    parser.add_argument('--speak', action='store_true', help='Enable Speak Mode')
    parser.add_argument('--debug', action='store_true', help='Enable Debug Mode')
    parser.add_argument('--gpt3only', action='store_true', help='Enable GPT3.5 Only Mode')
    args = parser.parse_args()

    if args.continuous:
        print_to_console("Continuous Mode: ", Fore.RED, "ENABLED")
        print_to_console(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorise. Use at your own risk.")
        cfg.set_continuous_mode(True)

    if args.speak:
        print_to_console("Speak Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_speak_mode(True)

    if args.gpt3only:
        print_to_console("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_smart_llm_model(cfg.fast_llm_model)

    if args.debug:
        print_to_console("Debug Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_debug_mode(True)