import random
import string
import sys


def generate_password(length: int = 8) -> str:
    if length < 8 or length > 16:
        raise ValueError("Password length must be between 8 and 16 characters.")

    characters = string.ascii_letters + string.digits + string.punctuation
    password = [
        random.choice(string.ascii_lowercase),
        random.choice(string.ascii_uppercase),
        random.choice(string.digits),
        random.choice(string.punctuation),
    ]
    password += [random.choice(characters) for _ in range(length - 4)]
    random.shuffle(password)
    return "".join(password)


if __name__ == "__main__":
    password_length = (
        int(sys.argv[sys.argv.index("--length") + 1]) if "--length" in sys.argv else 8
    )
    print(generate_password(password_length))
