import requests

BASE_URL = 'http://127.0.0.1:8080/ap/v1/'

# Input
data = {
    "input": "Create a three_sum function in a file called sample_code.py. Given an array of integers, return indices of the three numbers such that they add up to a specific target. You may assume that each input would have exactly one solution, and you may not use the same element twice. Example: Given nums = [2, 7, 11, 15], target = 20, Because nums[0] + nums[1] + nums[2] = 2 + 7 + 11 = 20, return [0, 1, 2].",
    "eval_id": "a1ff38a4-1032-4bf2-960a-3b927f9936f4"
}

data2 = {
    "input": "Read the file called file_to_read.txt and write its content to a file called output.txt",
    "eval_id": "f219f3d3-a41b-45a9-a3d0-389832086ee8"
}

data3 = {
    "input": "Create a random password generator. The password should have between 8 and 16 characters and should contain letters, numbers and symbols. "
             "The password should be printed to the console."
             "The entry point will be a python file that can be run this way: python password_generator.py [--len x] "
             "where x is the length of the password. If no length is specified, the password should be 8 characters long. "
             "The password_generator can also be imported as a module and called as password = password_generator.generate_password(len=x). "
             "Any invalid input should raise a ValueError.",
    "eval_id": "ac75c471-e0ce-400c-ba9a-fb72aaab444f"
}
# Step 1: POST request to /agent/tasks
response = requests.post(BASE_URL + 'agent/tasks', json=data3)
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
    if step["is_last"]:
        finished = True

# Step 3: POST request to /agent/tasks/{task_id}/evaluations
eval_endpoint = f'agent/tasks/{task_id}/evaluations'
response_eval = requests.post(BASE_URL + eval_endpoint)
if response_eval.status_code != 200:
    print(f"Failed to make the request to {eval_endpoint}")
    exit()

print("All requests completed successfully!")