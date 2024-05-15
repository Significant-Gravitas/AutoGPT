import random
import string

def generate_strong_password(length: int) -> str:
    if length < 10 or length > 20:
        raise ValueError("Password length must be between 10 and 20 characters.")
    
    characters = string.ascii_letters + string.digits + string.punctuation
    password = [
        random.choice(string.ascii_lowercase) for _ in range(2) +
        random.choice(string.ascii_uppercase) for _ in range(2) +
        random.choice(string.digits) for _ in range(2) +
        random.choice(string.punctuation) for _ in range(2)
    ]
    if length > 8:
        password += [random.choice(characters) for _ in range(length - 8)]
    random.shuffle(password)
    return "".join(password)

if __name__ == "__main__":
    password_length = int(input("Enter the length of the password (between 10 and 20): "))
    print(generate_strong_password(password_length))
