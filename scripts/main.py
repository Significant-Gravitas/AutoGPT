from auto_gpt import AutoGPT
import argparse


# For backwards compatibility with the old main.py
def main():
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument(
        "--continuous", action="store_true", help="Enable Continuous Mode"
    )
    parser.add_argument("--speak", action="store_true", help="Enable Speak Mode")
    parser.add_argument("--debug", action="store_true", help="Enable Debug Mode")
    parser.add_argument(
        "--gpt3only", action="store_true", help="Enable GPT3.5 Only Mode"
    )
    args = parser.parse_args()
    # Initialize AutoGPT
    auto_gpt = AutoGPT(
        continous_mode=args.continuous,
        speak_mode=args.speak,
        gpt3only_mode=args.gpt3only,
    )
    auto_gpt.run()


if __name__ == "__main__":
    main()
