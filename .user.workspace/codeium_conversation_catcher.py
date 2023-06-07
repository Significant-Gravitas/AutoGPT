import os
import re
import time

import pyautogui
import pytesseract
from PIL import Image
from chat_utils import activate_chat_panel

# Call the activate_chat_panel() function to activate the chat panel and bring the response window into focus
activate_chat_panel()

# Continue with the rest of your script



# Define the Capture Area for the Screenshot
x = 120  # left / -right coordinate of the capture area.
y = 150  #   up / -down coordinate of the capture area.
w = 1230 # width of the capture area.
h = 1660 # height of the capture area.

# Define the Delay Between Screenshot Captures
delay = 1


# PyAutoGUI key library
def press_key(key):
    pyautogui.press(key)


# Capture a Screenshot Region and Return it.
def take_screenshot(x, y, w, h):
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    return screenshot


# Activate the Chat Panel and focus on Response Window.
def activate_chat_panel():
    pyautogui.hotkey("ctrl", "shift", "a")
    time.sleep(2)
    pyautogui.click(x + w // 2, y + h // 4)

# Capture a Screenshot Region and Return it.
def capture_screenshot(width):
    screenshots = [take_screenshot(x, y, width, h)]

    _extracted_from_capture_screenshot_1(screenshots)
    _extracted_from_capture_screenshot_1(screenshots)
    # Save the screenshots to disk
    for i, screenshot in enumerate(screenshots):
        screenshot.save(f"screenshot{i+1}.png")

    return screenshots


# TODO Rename this here and in `capture_screenshot`
def _extracted_from_capture_screenshot_1(screenshots):
    # Capture the second screenshot after scrolling up once with the pgup key
    time.sleep(delay)
    press_key("pgup")
    screenshots.append(take_screenshot(x, y, w, h))


def stitch_screenshots(screenshot1, screenshot2, screenshot3):
    chat_height = screenshot1.height
    response_height = screenshot2.height

    full_screenshot = Image.new("RGB", (w, chat_height + (2 * response_height)))
    full_screenshot.paste(screenshot1, (0, response_height))
    full_screenshot.paste(screenshot2, (0, 0))
    full_screenshot.paste(screenshot3, (0, chat_height + response_height))

    full_screenshot.save(os.path.join(os.getcwd(), "full_screenshot.png"))
    print("Stitched image saved to disk.\n")

    return full_screenshot


# Return to Previous Chat Panel View
def press_key_1(delay):
    time.sleep(delay)
    press_key("pgdn")
    press_key("pgdn")
    print("Previous view restored.\n")


# Extract Text From Image
def extract_text_from_image(image):
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)


def find_clear_conversation_text(data):
    clear_conversation_index = next(
        (i for i, word in enumerate(data["text"]) if word == "Clear" and data["text"][i + 1] == "conversation"),
        -1,
    )
    return (
        data["text"][clear_conversation_index] + " " + data["text"][clear_conversation_index + 1]
        if clear_conversation_index != -1
        else ""
    )


def find_timestamp(data, timestamp_regex):
    timestamp_index = next(
        (i for i in range(len(data["text"]) - 1, -1, -1) if re.match(timestamp_regex, data["text"][i])),
        -1,
    )
    return data["text"][timestamp_index] if timestamp_index != -1 else ""


def capture_response_text():
    # Capture the response window
    response_window = (1140, 320, 1740, 920)  # left, top, right, bottom
    response_image = pyautogui.screenshot(region=response_window)

    # Save the response image
    response_image.save("response.png")

    return pytesseract.image_to_string(Image.open("response.png"))


def write_to_file(text):
    with open("codeium_conversation.md", "w") as f:
        f.write(text)


def main():
    # This pattern matches timestamps in the format "Tue, Jun 6, 2023, 6:02 pm"
    timestamp_regex = r"\w{3}, \w{3} \d{1,2}, \d{4}, \d{1,2}:\d{2} [ap]m"

    # Activate the chat panel and wait for it to load
    activate_chat_panel()
    time.sleep(2)

    # Capture the initial screenshot
    try:
        screenshot1, screenshot2, screenshot3 = capture_screenshot(w)
    except Exception:
        print("OCR failed on initial screenshot. Trying extended final screenshot...")
        screenshot1, screenshot2, screenshot3 = capture_screenshot(w + 100)

    # Stitch the screenshots together
    full_screenshot = stitch_screenshots(screenshot1, screenshot2, screenshot3)

    # Extract text from the full screenshot
    data = extract_text_from_image(full_screenshot)

    # Find the "Clear conversation" text in the chat panel
    clear_conversation_text = find_clear_conversation_text(data)

    # Find the timestamp in the response window
    timestamp_text = find_timestamp(data, timestamp_regex)

    # Capture the response text
    try:
        response_text = capture_response_text()
    except Exception:
        response_text = "Failed to extract response text from image."

    # Combine the extracted text into a Markdown-formatted string
    text = f"""# Codeium Conversation

## Timestamp

{timestamp_text}

## Conversation

{response_text}

{clear_conversation_text}
"""

    # Write the text to a file
    write_to_file(text)


if __name__ == "__main__":
    main()
