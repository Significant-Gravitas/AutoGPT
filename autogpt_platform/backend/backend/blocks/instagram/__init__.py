"""
Instagram Bot blocks for AutoGPT Platform.

This module provides Instagram automation capabilities using the instagrapi library.
All blocks support Instagram authentication via credentials.

Note: Instagram blocks require instagrapi to be installed separately via pip.
If instagrapi is not available, the blocks will not be loaded.
"""

__all__ = []

try:
    from .auth import (
        InstagramCredentials,
        InstagramCredentialsField,
        InstagramCredentialsInput,
    )
    from .comment import InstagramCommentBlock
    from .follow import InstagramFollowUserBlock, InstagramUnfollowUserBlock
    from .like import InstagramLikePostBlock, InstagramUnlikePostBlock
    from .login import InstagramLoginBlock
    from .post import InstagramPostPhotoBlock, InstagramPostReelBlock
    from .search import InstagramGetUserInfoBlock, InstagramSearchHashtagBlock

    __all__ = [
        "InstagramCredentials",
        "InstagramCredentialsField",
        "InstagramCredentialsInput",
        "InstagramLoginBlock",
        "InstagramPostPhotoBlock",
        "InstagramPostReelBlock",
        "InstagramLikePostBlock",
        "InstagramUnlikePostBlock",
        "InstagramFollowUserBlock",
        "InstagramUnfollowUserBlock",
        "InstagramCommentBlock",
        "InstagramGetUserInfoBlock",
        "InstagramSearchHashtagBlock",
    ]
except ImportError:
    # instagrapi is not installed - Instagram blocks will not be available
    pass
