from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    Credentials,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import CreatePostRequest, PostResponse, PostStatus, create_post
from ._config import wordpress


class WordPressCreatePostBlock(Block):
    """
    Creates a new post on a WordPress.com site or Jetpack-enabled site and publishes it.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = wordpress.credentials_field()
        site: str = SchemaField(
            description="Site ID or domain (e.g., 'myblog.wordpress.com' or '123456789')"
        )
        title: str = SchemaField(description="The post title")
        content: str = SchemaField(description="The post content (HTML supported)")
        excerpt: str | None = SchemaField(
            description="An optional post excerpt/summary", default=None
        )
        slug: str | None = SchemaField(
            description="The URL slug for the post (auto-generated if not provided)",
            default=None,
        )
        author: str | None = SchemaField(
            description="Username or ID of the author (defaults to authenticated user)",
            default=None,
        )
        categories: list[str] = SchemaField(
            description="List of category names or IDs", default=[]
        )
        tags: list[str] = SchemaField(
            description="List of tag names or IDs", default=[]
        )
        featured_image: str | None = SchemaField(
            description="Post ID of an existing attachment to set as featured image",
            default=None,
        )
        media_urls: list[str] = SchemaField(
            description="URLs of images to sideload and attach to the post", default=[]
        )

    class Output(BlockSchema):
        post_id: int = SchemaField(description="The ID of the created post")
        post_url: str = SchemaField(description="The full URL of the created post")
        short_url: str = SchemaField(description="The shortened wp.me URL")
        post_data: dict = SchemaField(description="Complete post data returned by API")

    def __init__(self):
        super().__init__(
            id="ee4fe08c-18f9-442f-a985-235379b932e1",
            description="Create a new post on WordPress.com or Jetpack sites",
            categories={BlockCategory.SOCIAL},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: Credentials, **kwargs
    ) -> BlockOutput:
        post_request = CreatePostRequest(
            title=input_data.title,
            content=input_data.content,
            excerpt=input_data.excerpt,
            slug=input_data.slug,
            author=input_data.author,
            categories=input_data.categories,
            tags=input_data.tags,
            featured_image=input_data.featured_image,
            media_urls=input_data.media_urls,
            status=PostStatus.PUBLISH,
        )

        post_response: PostResponse = await create_post(
            credentials=credentials,
            site=input_data.site,
            post_data=post_request,
        )

        yield "post_id", post_response.ID
        yield "post_url", post_response.URL
        yield "short_url", post_response.short_URL
        yield "post_data", post_response.model_dump()
