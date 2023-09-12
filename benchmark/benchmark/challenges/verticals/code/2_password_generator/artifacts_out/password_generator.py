import random
import string


def generate_password(length: int) -> str:
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
    password_length = random.randint(8, 16)
    print(generate_password(password_length))
