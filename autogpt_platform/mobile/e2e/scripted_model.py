"""External model boundary for the real-server native chat journey."""

import argparse
import asyncio
import json
import time
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
REPLY = (
    "Your local workspace is connected. Tell me what you would like Otto "
    "and your experts to work on."
)


@app.post("/v1/chat/completions")
async def completion(request: Request):
    payload = await request.json()
    base = {
        "id": f"chatcmpl-{uuid4()}",
        "model": payload.get("model", "native-e2e"),
        "created": int(time.time()),
    }
    if not payload.get("stream"):
        return {
            **base,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Native conversation"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def events():
        for word in REPLY.split(" "):
            yield "data: " + json.dumps(
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None,
                        }
                    ],
                }
            ) + "\n\n"
            await asyncio.sleep(0.04)
        yield "data: " + json.dumps(
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 20,
                    "total_tokens": 21,
                },
            }
        ) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18810)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
