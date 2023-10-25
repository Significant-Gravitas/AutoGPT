import requests

BASE_URL = 'http://127.0.0.1:8080/ap/v1/'

data1 = {
    "input": "Write the word 'Washington' to a .txt file",
    "eval_id": "f219f3d3-a41b-45a9-a3d0-389832086ee8"
}

three_sum = "a1ff38a4-1032-4bf2-960a-3b927f9936f4"


data2 = {
    "input": "Read the file called file_to_read.txt and write its content to a file called output.txt",
    "eval_id": "f219f3d3-a41b-45a9-a3d0-389832086ee8"
}

password_generator = "ac75c471-e0ce-400c-ba9a-fb72aaab444f"

battleship ="4d613d05-475f-4f72-bf12-f6d3714340c1"


# Step 1: POST request to /agent/tasks
response = requests.post(BASE_URL + 'agent/tasks', json={"input": "input", "eval_id": password_generator})
if response.status_code != 200:
    print("Failed to make the request to /agent/tasks")
    exit()

# Extract task_id from the response
task_id = response.json().get("task_id")
if not task_id:
    print("Task ID not found in the response")
    exit()

# Step 2: POST request to /agent/tasks/{task_id}/steps
finished = False
steps_endpoint = f'agent/tasks/{task_id}/steps'
while not finished:
    response_steps = requests.post(BASE_URL + steps_endpoint, json={})
    if response_steps.status_code != 200:
        print(f"Failed to make the request to {steps_endpoint}")
        exit()

    step = response_steps.json()
    print(step)
    # wait for user to press enter
    print("Press enter to continue...")
    input()
    if step["is_last"]:
        finished = True

# Step 3: POST request to /agent/tasks/{task_id}/evaluations
eval_endpoint = f'agent/tasks/{task_id}/evaluations'
response_eval = requests.post(BASE_URL + eval_endpoint)
if response_eval.status_code != 200:
    print(f"Failed to make the request to {eval_endpoint}")
    exit()

print("All requests completed successfully!")