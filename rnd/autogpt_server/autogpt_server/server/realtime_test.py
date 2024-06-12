import pytest
import time

from autogpt_server.server.realtime import start_realtime, start_realtime_connector 
from autogpt_server.data.execution import ExecutionQueue, Execution  

@pytest.mark.asyncio
async def test_realtime():
    api_queue = ExecutionQueue()
    exec_queue = ExecutionQueue()
    
    start_realtime(api_queue, exec_queue)
    api_queue.add(Execution(run_id="1", node_id="2", data={}))
    time.sleep(0.1) 
    msg: Execution = api_queue.get()
    assert msg.node_id == "hi"
