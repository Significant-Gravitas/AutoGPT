"""
Instagram Bot blocks for AutoGPT Platform.

This module provides Instagram automation capabilities using the instagrapi library.
All blocks support Instagram authentication via credentials.
"""

from .auth import InstagramCredentials, InstagramCredentialsField, InstagramCredentialsInput
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
