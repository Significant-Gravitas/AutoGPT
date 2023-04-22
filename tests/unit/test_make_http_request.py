import unittest
from unittest.mock import patch
from requests import Response
from autogpt.commands.web_requests import (
    make_http_request,
    is_valid_url,
    sanitize_url,
    get_response,
)


class TestWebRequests(unittest.TestCase):
    def test_is_valid_url(self):
        self.assertTrue(is_valid_url("https://www.example.com"))
        self.assertFalse(is_valid_url("not_a_valid_url"))

    def test_sanitize_url(self):
        self.assertEqual(
            sanitize_url("https://www.example.com/some_page"),
            "https://www.example.com/some_page",
        )

    @patch("autogpt.commands.web_requests.requests.request")
    def test_make_http_request(self, mock_request):
        url = "https://www.example.com"
        method = "GET"
        response = Response()
        response.status_code = 200
        mock_request.return_value = response

        result = make_http_request(url, method)
        self.assertEqual(result, response)
        mock_request.assert_called_once_with(method, url, json=None, headers={})

    def test_get_response(self):
        with patch("autogpt.commands.web_requests.session.get") as mock_get:
            mock_get.return_value = Response()
            mock_get.return_value.status_code = 200
            url = "https://www.example.com"
            response, error_message = get_response(url)
            self.assertIsNotNone(response)
            self.assertIsNone(error_message)

if __name__ == "__main__":
    unittest.main()
