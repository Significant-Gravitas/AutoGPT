import time
import os
import random
import string
import livekit
import subprocess
from multiprocessing import Process
from autogpt_server.data.execution import ExecutionQueue, Execution

# Reminder of types of todo notes I can add to the file...
# PERF: Performance
# HACK: Hack
# TODO: Todo
# NOTE: Note
# FIX: Fix me
# WARNING: Warning


def start_realtime(api_queue: ExecutionQueue, execution_queue: ExecutionQueue) -> bool:
    """
    Starts the livekit server and the realtime connector
    """

    # Start the livekit server
    lksp = subprocess.Popen(
        ["livekit-server", "--dev"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    # Start the connector
    realtime_process = Process(
        target=start_realtime_connector, args=(api_queue, execution_queue)
    )
    realtime_process.start()
    return True


async def start_realtime_connector(
    api_queue: ExecutionQueue, execution_queue: ExecutionQueue
) -> None:
    """
    This needs to startup the connector.

    The connector needs to manage the livekit-server in response to events
    that come from the agent server-api, handle livekit data streams and
    send events to the executor manager.
    """

    # FIX: Comms channels currently they do not appear to be working
    # TODO: Define API Event types that the realtime system can consume.
    # TODO: Define the tirggering event trypes that can be sent to the executor
    api_queue.add(Execution(run_id="1", node_id="connector Started", data={}))
    while True:
        amsg = api_queue.get()
        if amsg:
            print(amsg)
            api_queue.add(Execution(run_id="1", node_id="hi", data={}))
        emsg = execution_queue.get()
        if emsg:
            print(emsg)
        time.sleep(0.1)


# TODO: Add function to setup a new room
def setup_new_room(agent_id: str, timeout: int = 60) -> dict:
    """
    Sets up a new room on the LiveKit server using a random name based on the agent_id.

    Args:
        agent_id (str): The agent's unique identifier.
        timeout (int, optional): Timeout for the room in seconds. Defaults to 60.

    Returns:
        dict: Details of the created room or error information.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Generate a random room name using the agent_id
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    room_name = f"{agent_id}-{random_suffix}"

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Create a new room
        room_options = {"name": room_name, "timeout": timeout}
        room = client.create_room(room_options)
        return {"room": room}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to join agents room
def join_agents_room(agent_id: str, room_name: str) -> dict:
    """
    Joins an existing room for the agent.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room to join.

    Returns:
        dict: Response indicating the success or failure of joining the room.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Join the room
        response = client.join_room(room_name, agent_id)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to leave agents room
def leave_agents_room(agent_id: str, room_name: str) -> dict:
    """
    Leaves an existing room for the agent.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room to leave.

    Returns:
        dict: Response indicating the success or failure of leaving the room.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Leave the room
        response = client.leave_room(room_name, agent_id)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to get room details
def get_room_details(agent_id: str, room_name: str) -> dict:
    """
    Gets the details of an existing room.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room.

    Returns:
        dict: Details of the room or error information.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Get room details
        room_details = client.get_room(room_name)
        return {"room_details": room_details}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to to delete room
def delete_room(agent_id: str, room_name: str) -> dict:
    """
    Deletes an existing room.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room to delete.

    Returns:
        dict: Response indicating the success or failure of deleting the room.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Delete the room
        response = client.delete_room(room_name)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to listen to room
def listen_to_room(agent_id: str, room_name: str) -> dict:
    """
    Listens to the events in the room.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room to listen to.

    Returns:
        dict: Events from the room or error information.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Listen to room events
        events = client.listen_room_events(room_name)
        return {"events": events}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to process audio
def process_audio(agent_id: str, audio_stream) -> dict:
    """
    Processes audio from the audio stream.

    Args:
        agent_id (str): The agent's unique identifier.
        audio_stream: The audio stream to process.

    Returns:
        dict: Processed audio data or error information.
    """
    try:
        # TODO: Implement audio processing logic
        processed_audio = {"agent_id": agent_id, "processed_audio": audio_stream}
        return {"processed_audio": processed_audio}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to process video
def process_video(agent_id: str, video_stream) -> dict:
    """
    Processes video from the video stream.

    Args:
        agent_id (str): The agent's unique identifier.
        video_stream: The video stream to process.

    Returns:
        dict: Processed video data or error information.
    """
    try:
        # TODO: Implement video processing logic
        processed_video = {"agent_id": agent_id, "processed_video": video_stream}
        return {"processed_video": processed_video}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to process data channel, needs to be able to handle chat messages and files.
def process_data_channel(agent_id: str, data_channel) -> dict:
    """
    Processes data from the data channel, handling chat messages and files.

    Args:
        agent_id (str): The agent's unique identifier.
        data_channel: The data channel to process.

    Returns:
        dict: Processed data or error information.
    """
    try:
        processed_data = []
        # NOTE: Extend this logic to handle more data types as necessary.
        for data in data_channel:
            # TODO: Handle different types of data (e.g., chat messages, files)
            if data.get("type") == "chat":
                processed_data.append({"chat_message": data["content"]})
            elif data.get("type") == "file":
                processed_data.append({"file": data["content"]})
            else:
                processed_data.append({"unknown_data": data})

        return {"processed_data": processed_data}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to send audio
def send_audio(agent_id: str, room_name: str, audio_data) -> dict:
    """
    Sends audio data to the specified room.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room to send audio to.
        audio_data: The audio data to send.

    Returns:
        dict: Response from sending the audio or error information.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Send audio data
        response = client.send_audio(room_name, agent_id, audio_data)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to send video, possibly needs to be combined with audio?
def send_video(agent_id: str, room_name: str, video_data, audio_data=None) -> dict:
    """
    Sends video data (optionally combined with audio) to the specified room.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room to send video to.
        video_data: The video data to send.
        audio_data: The audio data to send along with video (optional).

    Returns:
        dict: Response from sending the video (and audio) or error information.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Send video data (and audio if provided)
        if audio_data:
            response = client.send_av(room_name, agent_id, video_data, audio_data)
        else:
            response = client.send_video(room_name, agent_id, video_data)

        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


# TODO: Add function to send data - messages and file
def send_data(agent_id: str, room_name: str, data, data_type: str) -> dict:
    """
    Sends data to the specified room. Handles chat messages and files.

    Args:
        agent_id (str): The agent's unique identifier.
        room_name (str): The name of the room to send data to.
        data: The data to send (e.g., message content, file content).
        data_type (str): The type of data being sent ('chat' or 'file').

    Returns:
        dict: Response from sending the data or error information.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        return {"error": "API key or secret not found in environment variables"}

    # Initialize LiveKit client
    client = livekit.Client(
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        if data_type == "chat":
            response = client.send_chat_message(room_name, agent_id, data)
        elif data_type == "file":
            response = client.send_file(room_name, agent_id, data)
        else:
            return {"error": "Unsupported data type"}

        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
