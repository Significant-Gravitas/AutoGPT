import pyautogui
import time

# Define the Capture Area for the Screenshot
x = 120  # left / -right coordinate of the capture area.
y = 150  #   up / -down coordinate of the capture area.
w = 1230 # width of the capture area.
h = 1660 # height of the capture area.


def activate_chat_panel():
    pyautogui.hotkey("ctrl", "shift", "a")
    time.sleep(2)
    pyautogui.click(x + w // 2, y + h // 4)


# Capture a Screenshot Region and Return it.
def take_screenshot(x, y, w, h):
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    return screenshot


# Capture a Screenshot Region and Return it.
def capture_screenshot(width):
    screenshots = [take_screenshot(x, y, width, h)]

    _extracted_from_capture_screenshot_1(screenshots)
    _extracted_from_capture_screenshot_1(screenshots)
    # Save the screenshots to disk
    for i, screenshot in enumerate(screenshots):
        screenshot.save(f"screenshot{i+1}.png")

    return screenshots


# Rename this function in both this file and in the call to it in main()
def _extracted_from_capture_screenshot_1(screenshots):
    # Capture the second screenshot after scrolling up once
    pyautogui.scroll(-h // 2)
    time.sleep(0.5)
    screenshots.append(take_screenshot(x, y, w, h))
