import argparse
import base64

URL_MAPPING = {}


def shorten_url(url):
    # Convert the URL to base64
    encoded_url = base64.b64encode(url.encode()).decode()
    # Take the first 8 characters of the encoded URL as our shortened URL
    short_url = encoded_url[:8]
    # Map the shortened URL back to the original
    URL_MAPPING[short_url] = url
    return short_url


def retrieve_url(short_url):
    return URL_MAPPING.get(short_url, "URL not found")


def main():
    parser = argparse.ArgumentParser(description="URL Shortener")
    parser.add_argument("-s", "--shorten", type=str, help="URL to be shortened")
    parser.add_argument("-r", "--retrieve", type=str, help="Short URL to be retrieved")

    args = parser.parse_args()

    if args.shorten:
        shortened_url = shorten_url(args.shorten)
        print(shortened_url)
        # Directly retrieve after shortening, using the newly shortened URL
        print(retrieve_url(shortened_url))
    elif args.retrieve:
        print(retrieve_url(args.retrieve))
    else:
        print("No valid arguments provided.")


if __name__ == "__main__":
    main()
