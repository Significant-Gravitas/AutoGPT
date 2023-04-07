import asyncio
import ctypes
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from scripts.file_operations import write_to_file, read_file
sys.path.append("scripts/")
from scripts.main import construct_prompt, Agent, ActiveAgent
from starlette.responses import JSONResponse
from typing import Dict
from fastapi import APIRouter
from scripts.ai_config import Agent
from scripts.config import Config
from scripts.memory import PineconeMemory, get_memory
import threading

router = APIRouter()

thread_dict = {}  # Dictionary to store the threads


@router.post("/agents")
async def create_agent(body: Dict): # TODO pydantify this when we have a clear definition of the request pattern.
    ai_config = Agent(
        ai_name=body["ai_name"],
        ai_role=body["ai_role"],
        ai_goals=["ai_goals"])
    ai_config.save()

    return JSONResponse(
        content={"message": "Agent created", "id": 1})  # only one main agent is supported for now

@router.patch("/agents/{id}")
async def patch_agent(body: Dict): # TODO pydantify this when we have a clear definition of the request pattern.
    if body["status"] == 'active':  # this is the only status that is supported for now
        cfg = Config()
        cfg.memory_backend = "no_memory"
        cfg.set_continuous_mode(True)  # this endpoint only supports continuous mode for now
        memory = get_memory(cfg, init=True)

        prompt = construct_prompt(interactive_input=False)
        activeAgent = ActiveAgent(
            cfg=cfg,
            prompt=prompt,
            memory=memory,
            user_input="Determine which next command to use, and respond using the format specified above:"
        )

        # Start the loop in a separate thread
        thread = threading.Thread(target=activeAgent.start_loop)
        try:
            thread.start()
        except SystemExit:
            # when the agent shuts down after task completion, it returns a SystemExit exception
            pass

        # Get the thread ID
        thread_id = thread.ident
        thread_dict[thread_id] = thread  # Store the thread in the dictionary
        return JSONResponse(content={"message": "agent created", "thread_id": thread_id})
    elif body["status"] == 'inactive':
        if "thread_id" not in body:
            return JSONResponse(content={"message": "thread_id is required in the request body"}, status_code=400)
        thread_id = body["thread_id"]
        # Terminate the thread using the thread ID
        thread = thread_dict.get(thread_id)  # Get the thread from the dictionary
        if thread:
            if not thread.is_alive():
                return JSONResponse(content={"message": "The thread is already terminated"}, status_code=500)

            threads_to_terminate = read_file("threads_to_terminate.json")
            try:
                threads_to_terminate = json.loads(threads_to_terminate)
            except:
                threads_to_terminate = {}
            threads_to_terminate[str(thread_id)] = True

            write_to_file("threads_to_terminate.json", json.dumps(threads_to_terminate))
            await wait_for_thread(thread, 30)
            if thread.is_alive():  # if a thread cannot be terminated within 30 seconds, it is terminated forcefully.
                if not ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread.ident), ctypes.py_object(SystemError)
                ):
                    return JSONResponse(content={"message": "Thread termination failed, please try again."},
                                        status_code=500)
                del thread_dict[thread_id]
                return JSONResponse(content={"message": "agent terminated forcefully."})
            else:
                return JSONResponse(content={"message": "agent terminated gracefully."})
        else:
            return JSONResponse(content={"message": "No thread found with the given thread_id"}, status_code=404)

async def wait_for_thread(thread, timeout):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()

    await loop.run_in_executor(executor, thread.join, timeout)
