import uvicorn

from autogpt_server.server.ws_api import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
