def clean_input(prompt: str=''):
    try:
        return input(prompt)
    except KeyboardInterrupt:
        exit(0)

