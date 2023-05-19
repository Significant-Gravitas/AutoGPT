# Slack Bot for Auto GPT

1. Create Slack Bot.
2. Add keys to Auto GPT `.env` file.
    ```
    SLACK_SIGNING_SECRET=
    SLACK_BOT_TOKEN=
    ```
3. Run server at ai platform 172.16.201.33
    ```
    nohup uvicorn app:app --host 0.0.0.0 --port 30207 --reload &
    ```