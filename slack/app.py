from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from slack_sdk import WebClient
import os
import random
import string
import datetime
from dotenv import load_dotenv
import time
import hmac
import hashlib
import re
import json
import subprocess
load_dotenv('../.env')

app = FastAPI()
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

def prepare(content):
    content = re.sub(r'<@U[A-Z0-9]+>', '', content)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    length = 5
    random_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    folder_name = date_str + "_" + random_str
    folder = os.path.join(os.getcwd(), 'auto_gpt_workspace', folder_name)
    os.makedirs(folder, exist_ok=True)
    ai_settings = f"""ai_name: AutoAskup
ai_role: an AI that achieves below goals.
ai_goals:
- {content}
- Terminate if above goal is achieved.
api_budget: 3"""
    with open(os.path.join(folder, "ai_settings.yaml"), "w") as f:
        f.write(ai_settings)
    return folder


def format_output(bytes):
    text = bytes.decode()
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    text = text.strip()
    index = text.find('THOUGHTS')
    if index != -1:
        return text[index:]
    if "Thinking..." in text:
        return
    if text.startswith(('REASONING', 'PLAN', '- ', 'CRITICISM', 'SPEAK', 'NEXT ACTION', 'SYSTEM')):
        return text

def run_autogpt_slack(data):
    main_dir = os.path.dirname(os.getcwd())
    print('main_dir', main_dir)
    folder = prepare(data['event']['text'])
    print('folder', folder)
    process = subprocess.Popen(
        ["python", os.path.join(main_dir, 'slack', 'api.py'), os.path.join(main_dir, folder)],
        cwd=main_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    started_loop = False
    messages = []
    while True:
        output = process.stdout.readline()
        if (not output) and process.poll() is not None:
            break
        if output:
            print(output.decode())
            output = format_output(output)
            if output is None:
                continue
            if output.startswith('THOUGHTS'):
                started_loop = True
            if not started_loop:
                continue
            messages.append(output)
            if started_loop and output.startswith(('NEXT ACTION', 'SYSTEM')):
                client.chat_postMessage(
                    channel=data['event']['channel'],
                    text="\n".join(messages),
                    thread_ts=data['event']['ts']
                )
                messages = []
        rc = process.poll()
    for line in process.stderr:
        print(line.decode().strip())

    for fname in os.listdir(folder):
        if fname not in ['ai_settings.yaml', 'auto-gpt.json', 'file_logger.txt']:
            file = os.path.join(folder, fname)
            upload_text_file = client.files_upload(
                channels=data['event']['channel'],
                thread_ts=data['event']['ts'],
                title=fname,
                file=file,
            )

@app.post("/")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    # Get the request body and headers
    body = await request.body()
    headers = request.headers
    print('BODY', body)
    print('HEADER', headers)
    # if body.challenge:
    #     return JSONResponse(content=body.challenge)

    # Get the timestamp and signature from the headers
    timestamp = headers.get('X-Slack-Request-Timestamp')
    slack_signature = headers.get('X-Slack-Signature')

    # Avoid replay attacks
    if abs(time.time() - int(timestamp)) > 60 * 5:
        # The request timestamp is more than five minutes from local time.
        # It could be a replay attack, so let's ignore it.
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    # Create a basestring by concatenating the version, the request timestamp, and the request body
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"

    # Hash the basestring using your signing secret, take the hex digest, and prefix with the version number
    my_signature = 'v0=' + hmac.new(
        bytes(os.getenv('SLACK_SIGNING_SECRET'), 'utf-8'),
        msg=bytes(basestring, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()

    if hmac.compare_digest(my_signature, slack_signature):
        # The signatures match, so the request is from Slack
        # Process the request here...
        data = json.loads(body)
        background_tasks.add_task(run_autogpt_slack, data)
        client.chat_postMessage(
            channel=data['event']['channel'],
            text="AutoGPT가 실행됩니다.",
            thread_ts=data['event']['ts']
        )
        return JSONResponse(content="Launched AutoGPT.")
    else:
        # The signatures don't match, so the request is not from Slack
        raise HTTPException(status_code=401, detail="Invalid signature")

@app.get("/")
async def index():
    return 'AutoAskUp'

# nohup uvicorn app:app --host 0.0.0.0 --port 30207 --reload &