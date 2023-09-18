import threading
import time
import unittest

import requests


class TestAPIRequests(unittest.TestCase):
    URL = "http://localhost:8080"

    def test_post_correct_then_incorrect_test_name(self):
        payload1 = {"test": "WriteFile", "mock": True}

        # First POST request
        response1 = requests.post(self.URL + "/reports", json=payload1)
        self.assertEqual(response1.status_code, 200)
        # Here you might want to check other aspects of the response, e.g., response1.json()
        print(response1.json())
        self.assertNotEqual(response1.json()["tests"], {})
        payload2 = {"test": "TestWriteFile", "mock": True}

        # Second POST request
        response2 = requests.post(self.URL + "/reports", json=payload2)
        print(response2.json())

        self.assertEqual(response2.json()["tests"], {})
        assert response1.json() != {}
        # Here you might want to check other aspects of the response, e.g., response2.json()

    def test_invalid_payload(self):
        invalid_payload = {"invalid_key": "value"}
        response = requests.post(self.URL + "/reports", json=invalid_payload)
        self.assertEqual(response.status_code, 422)  # Assuming 400 for Bad Request

    def test_post_report_and_poll_updates(self):
        payload1 = {"test": "WriteFile", "mock": True}
        last_update_time = int(time.time())
        # First POST request in a separate thread
        threading.Thread(target=self.send_post_request, args=(payload1,)).start()

        # Give a short time to ensure POST request is initiated before GET requests start

        # Start GET requests
        for _ in range(5):
            # get the current UNIX time
            response = requests.get(
                f"{self.URL}/updates?last_update_time={last_update_time}"
            )
            if response.status_code == 200 and response.json():
                print("Received a non-empty response:", response.json())
                break

            time.sleep(1)  # wait for 1 second before the next request
        else:
            self.fail("No updates received")

    def send_post_request(self, payload):
        response = requests.post(f"{self.URL}/reports", json=payload)
        if response.status_code == 200:
            print(response.json())


if __name__ == "__main__":
    unittest.main()
