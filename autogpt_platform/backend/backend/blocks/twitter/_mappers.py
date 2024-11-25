# -------------- Tweets -----------------

# Tweet Expansions
EXPANSION_FRONTEND_TO_BACKEND_MAPPING = {
    "attachments_poll_ids": "attachments.poll_ids",
    "attachments_media_keys": "attachments.media_keys",
    "author_id": "author_id",
    "edit_history_tweet_ids": "edit_history_tweet_ids",
    "entities_mentions_username": "entities.mentions.username",
    "geo_place_id": "geo.place_id",
    "in_reply_to_user_id": "in_reply_to_user_id",
    "referenced_tweets_id": "referenced_tweets.id",
    "referenced_tweets_id_author_id": "referenced_tweets.id.author_id",
}


def get_backend_expansion(frontend_key: str) -> str:
    result = EXPANSION_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid expansion key: {frontend_key}")
    return result


# TweetReplySettings
REPLY_SETTINGS_FRONTEND_TO_BACKEND_MAPPING = {
    "mentioned_users": "mentionedUsers",
    "following": "following",
    "all_users": "all",
}


# TweetUserFields
def get_backend_reply_setting(frontend_key: str) -> str:
    result = REPLY_SETTINGS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid reply setting key: {frontend_key}")
    return result


USER_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "created_at": "created_at",
    "description": "description",
    "entities": "entities",
    "id": "id",
    "location": "location",
    "most_recent_tweet_id": "most_recent_tweet_id",
    "name_user": "name",
    "pinned_tweet_id": "pinned_tweet_id",
    "profile_image_url": "profile_image_url",
    "protected": "protected",
    "public_metrics": "public_metrics",
    "url": "url",
    "username": "username",
    "verified": "verified",
    "verified_type": "verified_type",
    "withheld": "withheld",
}


def get_backend_user_field(frontend_key: str) -> str:
    result = USER_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid user field key: {frontend_key}")
    return result


# TweetFields
FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "attachments": "attachments",
    "author_id": "author_id",
    "context_annotations": "context_annotations",
    "conversation_id": "conversation_id",
    "created_at": "created_at",
    "edit_controls": "edit_controls",
    "entities": "entities",
    "geo": "geo",
    "id": "id",
    "in_reply_to_user_id": "in_reply_to_user_id",
    "lang": "lang",
    "public_metrics": "public_metrics",
    "possibly_sensitive": "possibly_sensitive",
    "referenced_tweets": "referenced_tweets",
    "reply_settings": "reply_settings",
    "source": "source",
    "text": "text",
    "withheld": "withheld",
}


def get_backend_field(frontend_key: str) -> str:
    result = FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid field key: {frontend_key}")
    return result


# TweetPollFields
POLL_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "duration_minutes": "duration_minutes",
    "end_datetime": "end_datetime",
    "id": "id",
    "options": "options",
    "voting_status": "voting_status",
}


def get_backend_poll_field(frontend_key: str) -> str:
    result = POLL_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid poll field key: {frontend_key}")
    return result


PLACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "contained_within": "contained_within",
    "country": "country",
    "country_code": "country_code",
    "full_name": "full_name",
    "geo": "geo",
    "id": "id",
    "place_name": "name",
    "place_type": "place_type",
}


def get_backend_place_field(frontend_key: str) -> str:
    result = PLACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid place field key: {frontend_key}")
    return result


# TweetMediaFields
MEDIA_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "duration_ms": "duration_ms",
    "height": "height",
    "media_key": "media_key",
    "preview_image_url": "preview_image_url",
    "type": "type",
    "url": "url",
    "width": "width",
    "public_metrics": "public_metrics",
    "non_public_metrics": "non_public_metrics",
    "organic_metrics": "organic_metrics",
    "promoted_metrics": "promoted_metrics",
    "alt_text": "alt_text",
    "variants": "variants",
}


def get_backend_media_field(frontend_key: str) -> str:
    result = MEDIA_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid media field key: {frontend_key}")
    return result


# -------------- Spaces -----------------

# SpaceExpansions
EXPANSION_FRONTEND_TO_BACKEND_MAPPING_SPACE = {
    "invited_user_ids": "invited_user_ids",
    "speaker_ids": "speaker_ids",
    "creator_id": "creator_id",
    "host_ids": "host_ids",
    "topic_ids": "topic_ids",
}


def get_backend_space_expansion(frontend_key: str) -> str:
    result = EXPANSION_FRONTEND_TO_BACKEND_MAPPING_SPACE.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid expansion key: {frontend_key}")
    return result


# SpaceFields
SPACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "id": "id",
    "state": "state",
    "created_at": "created_at",
    "ended_at": "ended_at",
    "host_ids": "host_ids",
    "lang": "lang",
    "is_ticketed": "is_ticketed",
    "invited_user_ids": "invited_user_ids",
    "participant_count": "participant_count",
    "subscriber_count": "subscriber_count",
    "scheduled_start": "scheduled_start",
    "speaker_ids": "speaker_ids",
    "started_at": "started_at",
    "title_": "title",
    "topic_ids": "topic_ids",
    "updated_at": "updated_at",
}


def get_backend_space_field(frontend_key: str) -> str:
    result = SPACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid space field key: {frontend_key}")
    return result


# -------------- List Expansions -----------------

# ListExpansions
LIST_EXPANSION_FRONTEND_TO_BACKEND_MAPPING = {"owner_id": "owner_id"}


def get_backend_list_expansion(frontend_key: str) -> str:
    result = LIST_EXPANSION_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid list expansion key: {frontend_key}")
    return result


LIST_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "id": "id",
    "list_name": "name",
    "created_at": "created_at",
    "description": "description",
    "follower_count": "follower_count",
    "member_count": "member_count",
    "private": "private",
    "owner_id": "owner_id",
}


def get_backend_list_field(frontend_key: str) -> str:
    result = LIST_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid list field key: {frontend_key}")
    return result
