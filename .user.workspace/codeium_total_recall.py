import pyautogui
import time

# Define the output sections
SECTION_1 = {"name": "Standard AI language model", "input_file": "As_an_AI_language_model.md"}
SECTION_2 = {"name": "Custom AI language model", "input_file": "As_an_AI_language_model.md"}
SECTION_3 = {"name": "Try a different approach 1", "input_file": "As_an_AI_language_model.md"}

# Activate the Codeium chat panel
pyautogui.hotkey("ctrl", "shift", "a")
time.sleep(1)


# Communication with Codeium AI
def communicate_with_ai(message: str, current_section: dict) -> None:
    # Send a message to the Codeium chat panel
    try:
        print(f"Sending message to chat panel: {message}")
        pyautogui.typewrite(message)
        pyautogui.press("enter")
        # Wait for a response from the Codeium AI
        current_section = adjust_output_section(wait_for_response(), current_section)
        if not current_section:
            return
        # Output the relevant section of the AI's response
        with open(current_section["input_file"], "r") as f:
            input_text = f.read()
            output_start = input_text.find(current_section["name"])
            if output_start == -1:
                print("Error: section not found")
                return
            input_text = input_text[output_start:]
            output_end = input_text.find("#")
            if output_end == -1:
                output_end = len(input_text)
            output_text = input_text[:output_end].strip()
            print(f"Outputting section: {output_text}")
            communicate_with_ai(output_text, current_section)
    except Exception as e:
        print("An exception occurred:", e)


# Trigger Codeium AI memory
def trigger_memory() -> str:
    try:
        print("Executing memory trigger loop for Codeium AI")
        pyautogui.typewrite("Can you remind me what we talked about last session?")
        pyautogui.press("enter")
        return wait_for_response()
    except Exception as e:
        print("An exception occurred:", e)


# Wait for a response from the Codeium AI
def wait_for_response() -> str:
    print("Waiting for response from Codeium AI...")
    for _ in range(10):
        print("Locating Codeium AI response...")
        if response := pyautogui.locateOnScreen("codeium_response.png"):
            pyautogui.moveTo(response)
            pyautogui.move(50, 0)
            pyautogui.click()
            pyautogui.hotkey("ctrl", "a")
            pyautogui.hotkey("ctrl", "c")
            copied_text = pyautogui.paste()
            print(f"Copied text: {copied_text}")
            response = copied_text.strip()
            pyautogui.press("esc")
            return response
        time.sleep(1)
    return ""


# Adjust the output section based on the Codeium AI's response
def adjust_output_section(response: str, current_section: dict) -> dict:
    print(f"AI response: {response}, Output section: {current_section['name']}")
    if "You are correct" in response:
        return {}
    elif "I apologize, but as an AI language model, I don't have a memory" in response:
        try_different_approach()
        response = wait_for_response()

    if "Please provide more details about" in response:
        current_section = SECTION_2
    elif "How do I" in response:
        current_section = SECTION_3

    return current_section


# Try a different approach to trigger Codeium AI memory
def try_different_approach() -> None:
    try:
        print("Trying a different approach to trigger AI memory...")
        pyautogui.typewrite("Can you remind me what we talked about last time?")
        pyautogui.press("enter")
        time.sleep(1)
        pyautogui.hotkey("ctrl", "shift", "a")
        time.sleep(1)
        pyautogui.typewrite("Can you remind me what we talked about last session?")
        pyautogui.press("enter")
        wait_for_response()
    except Exception as e:
        print("An exception occurred:", e)


# Call the communicate_with_ai function to start the conversation
current_section = SECTION_1
communicate_with_ai("", current_section)
