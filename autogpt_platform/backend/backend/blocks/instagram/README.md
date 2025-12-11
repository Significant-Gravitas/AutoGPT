# Instagram Bot Blocks

Native Instagram automation blocks for AutoGPT Platform using the `instagrapi` library.

## Features

This module provides comprehensive Instagram automation capabilities:

### Authentication
- **InstagramLoginBlock** - Authenticate with Instagram using username/password

### Content Posting
- **InstagramPostPhotoBlock** - Post photos with captions and optional location
- **InstagramPostReelBlock** - Post video Reels with captions and thumbnails

### Engagement
- **InstagramLikePostBlock** - Like posts by media ID or URL
- **InstagramUnlikePostBlock** - Unlike posts
- **InstagramCommentBlock** - Comment on posts
- **InstagramFollowUserBlock** - Follow users by username or ID
- **InstagramUnfollowUserBlock** - Unfollow users

### Discovery
- **InstagramGetUserInfoBlock** - Get detailed user profile information
- **InstagramSearchHashtagBlock** - Search and retrieve posts by hashtag

## Setup

### 1. Install Dependencies

The `instagrapi` library must be installed **separately** due to dependency conflicts with the platform's core packages.

**Why separate installation?**
- `instagrapi` requires `moviepy==1.0.3` and specific `pydantic` versions
- AutoGPT platform requires `moviepy>=2.1.2` and `pydantic>=2.11.7`
- These constraints cannot be resolved together by Poetry

**Installation steps:**
```bash
cd autogpt_platform/backend

# 1. Install core platform dependencies
poetry install

# 2. Install instagrapi (will coexist with platform deps at runtime)
poetry run pip install instagrapi
```

**Note:** The Instagram blocks gracefully handle the case when `instagrapi` is not installed - they simply won't be loaded, and other blocks will continue to work normally.

### 2. Configure Credentials

Instagram credentials are stored in the format `username:password` as an API key.

**Example:**
- Username: `your_username`
- Password: `your_password`
- API Key: `your_username:your_password`

### 3. Add Credentials in AutoGPT Platform

1. Navigate to Settings → Integrations
2. Add new Instagram credentials
3. Provider: `instagram`
4. API Key: `your_username:your_password`

## Usage Examples

### Example 1: Post a Photo

```
1. InstagramPostPhotoBlock
   - photo_url: "https://example.com/image.jpg"
   - caption: "Check out this amazing view! #travel #photography"
   - location_name: "Paris, France"
```

### Example 2: Auto-Like Posts by Hashtag

```
1. InstagramSearchHashtagBlock
   - hashtag: "autogpt"
   - amount: 10

2. InstagramLikePostBlock (loop through post_urls)
   - media_id: {from search results}
```

### Example 3: Follow and Engage Workflow

```
1. InstagramGetUserInfoBlock
   - username: "target_user"

2. InstagramFollowUserBlock
   - username: {from user info}

3. InstagramSearchHashtagBlock
   - hashtag: "ai"
   - amount: 5

4. InstagramCommentBlock
   - media_id: {from search}
   - comment_text: "Great content! 🚀"
```

## Block Details

### InstagramPostPhotoBlock

**Inputs:**
- `photo_url` (required): URL or local path to photo
- `caption` (optional): Post caption (max 2,200 chars)
- `location_name` (optional): Location to tag

**Outputs:**
- `success`: Boolean indicating success
- `media_id`: Instagram media ID
- `media_code`: Short code for URL
- `post_url`: Full Instagram post URL

### InstagramSearchHashtagBlock

**Inputs:**
- `hashtag` (required): Hashtag to search (without #)
- `amount` (optional): Number of posts to retrieve (1-50, default: 10)

**Outputs:**
- `post_ids`: List of media IDs
- `post_urls`: List of post URLs
- `captions`: List of captions
- `like_counts`: List of like counts

### InstagramGetUserInfoBlock

**Inputs:**
- `username` (required): Username to look up

**Outputs:**
- `user_id`: User's Instagram ID
- `username`: Username
- `full_name`: Display name
- `biography`: Bio text
- `follower_count`: Number of followers
- `following_count`: Number of following
- `media_count`: Number of posts
- `is_private`: Whether account is private
- `is_verified`: Whether account is verified
- `profile_pic_url`: Profile picture URL

## Important Notes

### Rate Limiting

Instagram has strict rate limits. To avoid being flagged:

1. **Go Slow**: Add delays between actions (use Sleep block)
2. **Realistic Behavior**: Don't like/follow too many accounts rapidly
3. **Session Management**: Reuse sessions when possible
4. **Human-like Patterns**: Vary timing and actions

**Recommended limits per hour:**
- Likes: 60-100
- Follows: 20-40
- Comments: 20-30
- Posts: 3-5

### Best Practices

1. **Use a Test Account**: Test workflows on a non-primary account
2. **Add Error Handling**: Instagram actions can fail; handle errors gracefully
3. **Respect Privacy**: Only interact with public accounts
4. **Follow Terms of Service**: Ensure compliance with Instagram's TOS
5. **Use Delays**: Add 10-30 second delays between actions

### Security

- Never share your Instagram credentials
- Use strong, unique passwords
- Enable two-factor authentication on your Instagram account
- Credentials are stored securely using AutoGPT's credential system

## Troubleshooting

### "Invalid credentials format" Error

Ensure credentials are in `username:password` format:
```
✅ Correct: myuser:mypassword
❌ Wrong: myuser, mypassword
❌ Wrong: mypassword
```

### "Login failed" Error

- Verify username and password are correct
- Check if account requires 2FA (currently not supported)
- Instagram may be blocking automated logins - try from a different IP

### "Challenge required" Error

Instagram detected unusual activity:
- Wait 24-48 hours before retrying
- Use Instagram's mobile app to verify your identity
- Reduce automation frequency

### Media Not Found

- Verify the media ID or URL is correct
- Ensure the post is still available (not deleted)
- Check if the account is private and you're not following

## Contributing

To add new Instagram blocks:

1. Create a new file in the `instagram/` directory
2. Follow the existing block pattern
3. Add the block to `__init__.py`
4. Update this README with usage examples
5. Test thoroughly before submitting PR

## Dependencies

- `instagrapi` (^2.1.2): Instagram Private API wrapper

## License

This contribution falls under the AutoGPT Platform's Contributor License Agreement (CLA).
