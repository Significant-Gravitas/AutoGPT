import pyautogui


def get_response_window_coordinates():
    print("Please position your mouse cursor at the top-left corner of the response window and press a key.")
    pyautogui.pause = False
    pyautogui.hotkey("ctrl", "shift", "a")
    top_left = pyautogui.position()
    print(f"Top-left corner: {top_left}")

    print("Please position your mouse cursor at the bottom-right corner of the response window and press the same key.")
    bottom_right = pyautogui.position()
    print(f"Bottom-right corner: {bottom_right}")

    return (top_left, bottom_right)
