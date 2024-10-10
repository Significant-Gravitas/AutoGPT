from typing import List

import requests

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class PublishToMediumBlock(Block):
    class Input(BlockSchema):
        author_id: BlockSecret = SecretField(
            key="medium_author_id",
            description="""The Medium AuthorID of the user. You can get this by calling the /me endpoint of the Medium API.\n\ncurl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" https://api.medium.com/v1/me" the response will contain the authorId field.""",
            placeholder="Enter the author's Medium AuthorID",
        )
        title: str = SchemaField(
            description="The title of your Medium post",
            placeholder="Enter your post title",
        )
        content: str = SchemaField(
            description="The main content of your Medium post",
            placeholder="Enter your post content",
        )
        content_format: str = SchemaField(
            description="The format of the content: 'html' or 'markdown'",
            placeholder="html",
        )
        tags: List[str] = SchemaField(
            description="List of tags for your Medium post (up to 5)",
            placeholder="['technology', 'AI', 'blogging']",
        )
        canonical_url: str | None = SchemaField(
            default=None,
            description="The original home of this content, if it was originally published elsewhere",
            placeholder="https://yourblog.com/original-post",
        )
        publish_status: str = SchemaField(
            description="The publish status: 'public', 'draft', or 'unlisted'",
            placeholder="public",
        )
        license: str = SchemaField(
            default="all-rights-reserved",
            description="The license of the post: 'all-rights-reserved', 'cc-40-by', 'cc-40-by-sa', 'cc-40-by-nd', 'cc-40-by-nc', 'cc-40-by-nc-nd', 'cc-40-by-nc-sa', 'cc-40-zero', 'public-domain'",
            placeholder="all-rights-reserved",
        )
        notify_followers: bool = SchemaField(
            default=False,
            description="Whether to notify followers that the user has published",
            placeholder="False",
        )
        api_key: BlockSecret = SecretField(
            key="medium_api_key",
            description="""The API key for the Medium integration. You can get this from https://medium.com/me/settings/security and scrolling down to "integration Tokens".""",
            placeholder="Enter your Medium API key",
        )

    class Output(BlockSchema):
        post_id: str = SchemaField(description="The ID of the created Medium post")
        post_url: str = SchemaField(description="The URL of the created Medium post")
        published_at: int = SchemaField(
            description="The timestamp when the post was published"
        )
        error: str = SchemaField(
            description="Error message if the post creation failed"
        )

    def __init__(self):
        super().__init__(
            id="3f7b2dcb-4a78-4e3f-b0f1-88132e1b89df",
            input_schema=PublishToMediumBlock.Input,
            output_schema=PublishToMediumBlock.Output,
            description="Publishes a post to Medium.",
            categories={BlockCategory.SOCIAL},
            test_input={
                "author_id": "1234567890abcdef",
                "title": "Test Post",
                "content": "<h1>Test Content</h1><p>This is a test post.</p>",
                "content_format": "html",
                "tags": ["test", "automation"],
                "license": "all-rights-reserved",
                "notify_followers": False,
                "publish_status": "draft",
                "api_key": "your_test_api_key",
            },
            test_output=[
                ("post_id", "e6f36a"),
                ("post_url", "https://medium.com/@username/test-post-e6f36a"),
                ("published_at", 1626282600),
            ],
            test_mock={
                "create_post": lambda *args, **kwargs: {
                    "data": {
                        "id": "e6f36a",
                        "url": "https://medium.com/@username/test-post-e6f36a",
                        "authorId": "1234567890abcdef",
                        "publishedAt": 1626282600,
                    }
                }
            },
        )

    def create_post(
        self,
        api_key,
        author_id,
        title,
        content,
        content_format,
        tags,
        canonical_url,
        publish_status,
        license,
        notify_followers,
    ):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        data = {
            "title": title,
            "content": content,
            "contentFormat": content_format,
            "tags": tags,
            "canonicalUrl": canonical_url,
            "publishStatus": publish_status,
            "license": license,
            "notifyFollowers": notify_followers,
        }

        response = requests.post(
            f"https://api.medium.com/v1/users/{author_id}/posts",
            headers=headers,
            json=data,
        )

        return response.json()

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        response = self.create_post(
            input_data.api_key.get_secret_value(),
            input_data.author_id.get_secret_value(),
            input_data.title,
            input_data.content,
            input_data.content_format,
            input_data.tags,
            input_data.canonical_url,
            input_data.publish_status,
            input_data.license,
            input_data.notify_followers,
        )

        if "data" in response:
            yield "post_id", response["data"]["id"]
            yield "post_url", response["data"]["url"]
            yield "published_at", response["data"]["publishedAt"]
        else:
            error_message = response.get("errors", [{}])[0].get(
                "message", "Unknown error occurred"
            )
            raise RuntimeError(f"Failed to create Medium post: {error_message}")
