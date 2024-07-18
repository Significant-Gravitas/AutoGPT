import requests
from autogpt_server.data.block import Block, BlockSchema, BlockOutput, BlockFieldSecret

class CreateMediumPostBlock(Block):
    class Input(BlockSchema):
        author_id: str
        title: str
        content: str
        content_format: str = "html"
        tags: list[str] = []
        canonical_url: str = ""
        publish_status: str = "public"
        license: str = "all-rights-reserved"
        notify_followers: bool = False
        api_key: BlockFieldSecret = BlockFieldSecret(key="medium_api_key")

    class Output(BlockSchema):
        post_id: str
        post_url: str
        author_id: str
        published_at: int
        error: str

    def __init__(self):
        super().__init__(
            id="m3d1um-p0st-cr3a-t10n-bl0ck-1d3nt1f13r",
            input_schema=CreateMediumPostBlock.Input,
            output_schema=CreateMediumPostBlock.Output,
            test_input={
                "author_id": "5303d74c64f66366f00cb9b2a94f3251bf5",
                "title": "Test Post",
                "content": "<h1>Test Post</h1><p>This is a test post created by AutoGPT.</p>",
                "tags": ["test", "autogpt"],
                "api_key": "test-api-key"
            },
            test_output=("post_created", {
                "post_id": "e6f36a",
                "post_url": "https://medium.com/@username/test-post-e6f36a",
                "author_id": "5303d74c64f66366f00cb9b2a94f3251bf5",
                "published_at": 1442286338435
            }),
            test_mock={"create_post": lambda *args, **kwargs: {
                "data": {
                    "id": "e6f36a",
                    "title": "Test Post",
                    "authorId": "5303d74c64f66366f00cb9b2a94f3251bf5",
                    "url": "https://medium.com/@username/test-post-e6f36a",
                    "publishStatus": "public",
                    "publishedAt": 1442286338435,
                }
            }}
        )

    @staticmethod
    def create_post(author_id: str, api_key: str, post_data: dict) -> dict:
        url = f"https://api.medium.com/v1/users/{author_id}/posts"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Charset": "utf-8"
        }
        response = requests.post(url, json=post_data, headers=headers)
        response.raise_for_status()
        return response.json()

    def run(self, input_data: Input) -> BlockOutput:
        post_data = {
            "title": input_data.title,
            "contentFormat": input_data.content_format,
            "content": input_data.content,
            "tags": input_data.tags[:3],  # Medium only uses the first 3 tags
            "canonicalUrl": input_data.canonical_url,
            "publishStatus": input_data.publish_status,
            "license": input_data.license,
            "notifyFollowers": input_data.notify_followers
        }

        try:
            response = self.create_post(input_data.author_id, input_data.api_key.get(), post_data)
            post_data = response["data"]
            
            yield "post_id", post_data["id"],
            yield "post_url", post_data["url"],
            yield "author_id", post_data["authorId"],
            yield "published_at", post_data["publishedAt"],

        except requests.HTTPError as e:
            error_message = str(e)
            if e.response is not None:
                status_code = e.response.status_code
                if status_code == 400:
                    error_message = "Bad Request: The request was invalid. This could be due to missing required fields, invalid values, or an incorrect author ID."
                elif status_code == 401:
                    error_message = "Unauthorized: The access token is invalid or has been revoked."
                elif status_code == 403:
                    error_message = "Forbidden: The user does not have permission to publish, or the author ID is incorrect."
                
                # Try to get more details from the response
                try:
                    response_json = e.response.json()
                    if "errors" in response_json:
                        error_message += f" Details: {response_json['errors']}"
                except ValueError:
                    pass  # The response body wasn't valid JSON
            
            yield "error", error_message
        except requests.RequestException as e:
            yield "error", f"Network error occurred: {str(e)}"
        except Exception as e:
            yield "error", f"An unexpected error occurred: {str(e)}"