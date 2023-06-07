import pyautogui
import pytesseract


# Test the screenshot and OCR functionality
def test_ocr():
    # Take a screenshot of a sample chat panel
    x, y, w, h = 550, 50, 600, 700  # Replace with the coordinates of your chat panel
    screenshot = pyautogui.screenshot(region=(x + w - 600, y + h - 200, 600, 200))

    # Save the screenshot to a file (optional)
    screenshot.save('screenshot.png')

    # Convert the screenshot to text using OCR
    text = pytesseract.image_to_string(screenshot)

    # Check that the extracted text contains the expected message
    expected_message = "Hello, this is a test message."
    assert expected_message in text, f"Expected message '{expected_message}' not found in OCR output: '{text}'"


# Run the test
test_ocr()
