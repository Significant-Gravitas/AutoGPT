import argparse
import hashlib

HASH_MAPPING = {}


def hash_message(message):
    # Convert the message to a SHA-256 hash
    hashed_message = hashlib.sha256(message.encode()).hexdigest()
    # Map the hash back to the original message
    HASH_MAPPING[hashed_message] = message
    return hashed_message


def retrieve_message(hashed_message):
    return HASH_MAPPING.get(hashed_message, "Message not found")


def main():
    parser = argparse.ArgumentParser(description="Message Hasher")
    parser.add_argument("-h", "--hash", type=str, help="Message to be hashed")
    parser.add_argument("-v", "--verify", type=str, help="Hashed message to verify")

    args = parser.parse_args()

    if args.hash:
        hashed_message = hash_message(args.hash)
        print("Hashed Message:", hashed_message)
        # Directly retrieve after hashing, using the newly created hash
        print("Original Message:", retrieve_message(hashed_message))
    elif args.verify:
        print("Original Message:", retrieve_message(args.verify))
    else:
        print("No valid arguments provided.")


if __name__ == "__main__":
    main()
