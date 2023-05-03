import os
from unittest.mock import MagicMock, patch
import unittest

from scenex_plugin import describe_image, is_api_key_set, get_api_key

MOCK_API_KEY = "secret"
MOCK_IMAGE = "https://example.com/image.png"
MOCK_DESCRIPTION = "example description"


class TestEmailPlugin(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "SCENEX_API_KEY": MOCK_API_KEY,
        },
    )
    def test_api_key_set(self):
        self.assertTrue(is_api_key_set())
        self.assertEqual(get_api_key(), MOCK_API_KEY)

    @patch.dict(os.environ, {}, clear=True)
    def test_api_key_not_set(self):
        self.assertFalse(is_api_key_set())

    @patch.dict(
        os.environ,
        {
            "SCENEX_API_KEY": MOCK_API_KEY,
        },
    )
    @patch("scenex_plugin.requests.post")
    def test_describe_image(self, mock_post):
        mock_post.return_value = MagicMock(
            json=MagicMock(
                return_value={
                    "result": [
                        {
                            "image": MOCK_IMAGE,
                            "text": MOCK_DESCRIPTION,
                        }
                    ]
                }
            )
        )

        result = describe_image(
            image=MOCK_IMAGE,
            algorithm="Dune",
            features=[],
            languages=[],
        )

        # Check the results
        self.assertEqual(
            result,
            {
                "image": MOCK_IMAGE,
                "description": MOCK_DESCRIPTION,
            },
        )

        # Check that the mocked functions were called with the correct arguments
        mock_post.assert_called_once_with(
            "https://us-central1-causal-diffusion.cloudfunctions.net/describe",
            headers={
                "x-api-key": f"token {MOCK_API_KEY}",
                "content-type": "application/json",
            },
            json={
                "data": [
                    {
                        "image": MOCK_IMAGE,
                        "algorithm": "Dune",
                        "features": [],
                        "languages": [],
                    }
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
