from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict

from pydantic import BaseModel

from backend.blocks.twitter._mappers import (
    get_backend_expansion,
    get_backend_field,
    get_backend_list_expansion,
    get_backend_list_field,
    get_backend_media_field,
    get_backend_place_field,
    get_backend_poll_field,
    get_backend_space_expansion,
    get_backend_space_field,
    get_backend_user_field,
)
from backend.blocks.twitter._types import (  # DMEventFieldFilter,
    DMEventExpansionFilter,
    DMEventTypeFilter,
    DMMediaFieldFilter,
    DMTweetFieldFilter,
    ExpansionFilter,
    ListExpansionsFilter,
    ListFieldsFilter,
    SpaceExpansionsFilter,
    SpaceFieldsFilter,
    TweetFieldsFilter,
    TweetMediaFieldsFilter,
    TweetPlaceFieldsFilter,
    TweetPollFieldsFilter,
    TweetReplySettingsFilter,
    TweetUserFieldsFilter,
    UserExpansionsFilter,
)


# Common Builder
class TweetExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: ExpansionFilter | None):
        _add_selected(self.params, "expansions", expansions, get_backend_expansion)
        return self

    def add_media_fields(self, media_fields: TweetMediaFieldsFilter | None):
        _add_selected(
            self.params, "media.fields", media_fields, get_backend_media_field
        )
        return self

    def add_place_fields(self, place_fields: TweetPlaceFieldsFilter | None):
        _add_selected(
            self.params, "place.fields", place_fields, get_backend_place_field
        )
        return self

    def add_poll_fields(self, poll_fields: TweetPollFieldsFilter | None):
        _add_selected(self.params, "poll.fields", poll_fields, get_backend_poll_field)
        return self

    def add_tweet_fields(self, tweet_fields: TweetFieldsFilter | None):
        _add_selected(self.params, "tweet.fields", tweet_fields, get_backend_field)
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        _add_selected(self.params, "user.fields", user_fields, get_backend_user_field)
        return self

    def build(self):
        return self.params


class UserExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: UserExpansionsFilter | None):
        _add_selected(self.params, "expansions", expansions)
        return self

    def add_tweet_fields(self, tweet_fields: TweetFieldsFilter | None):
        _add_selected(self.params, "tweet.fields", tweet_fields, get_backend_field)
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        _add_selected(self.params, "user.fields", user_fields, get_backend_user_field)
        return self

    def build(self):
        return self.params


class ListExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: ListExpansionsFilter | None):
        _add_selected(self.params, "expansions", expansions, get_backend_list_expansion)
        return self

    def add_list_fields(self, list_fields: ListFieldsFilter | None):
        _add_selected(self.params, "list.fields", list_fields, get_backend_list_field)
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        _add_selected(self.params, "user.fields", user_fields, get_backend_user_field)
        return self

    def build(self):
        return self.params


class SpaceExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: SpaceExpansionsFilter | None):
        _add_selected(
            self.params, "expansions", expansions, get_backend_space_expansion
        )
        return self

    def add_space_fields(self, space_fields: SpaceFieldsFilter | None):
        _add_selected(
            self.params, "space.fields", space_fields, get_backend_space_field
        )
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        _add_selected(self.params, "user.fields", user_fields, get_backend_user_field)
        return self

    def build(self):
        return self.params


class TweetDurationBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_start_time(self, start_time: datetime | None):
        if start_time:
            # Twitter API requires start_time to be at least 10 seconds before now
            max_start_time = datetime.now(timezone.utc) - timedelta(seconds=10)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if start_time > max_start_time:
                start_time = max_start_time
            self.params["start_time"] = start_time
        return self

    def add_end_time(self, end_time: datetime | None):
        if end_time:
            self.params["end_time"] = end_time
        return self

    def add_since_id(self, since_id: str | None):
        if since_id:
            self.params["since_id"] = since_id
        return self

    def add_until_id(self, until_id: str | None):
        if until_id:
            self.params["until_id"] = until_id
        return self

    def add_sort_order(self, sort_order: str | None):
        if sort_order:
            self.params["sort_order"] = sort_order
        return self

    def build(self):
        return self.params


class DMExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: DMEventExpansionFilter):
        _add_selected(self.params, "expansions", expansions)
        return self

    def add_event_types(self, event_types: DMEventTypeFilter):
        _add_selected(self.params, "event_types", event_types)
        return self

    def add_media_fields(self, media_fields: DMMediaFieldFilter):
        _add_selected(self.params, "media.fields", media_fields)
        return self

    def add_tweet_fields(self, tweet_fields: DMTweetFieldFilter):
        _add_selected(self.params, "tweet.fields", tweet_fields)
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter):
        _add_selected(self.params, "user.fields", user_fields)
        return self

    def build(self):
        return self.params


# Specific Builders
class TweetSearchBuilder:
    def __init__(self):
        self.params: Dict[str, Any] = {"user_auth": False}

    def add_query(self, query: str):
        if query:
            self.params["query"] = query
        return self

    def add_pagination(self, max_results: int, pagination: str | None):
        if max_results:
            self.params["max_results"] = max_results
        if pagination:
            self.params["pagination_token"] = pagination
        return self

    def build(self):
        return self.params


class TweetPostBuilder:
    def __init__(self):
        self.params: Dict[str, Any] = {"user_auth": False}

    def add_text(self, text: str | None):
        if text:
            self.params["text"] = text
        return self

    def add_media(self, media_ids: list, tagged_user_ids: list):
        if media_ids:
            self.params["media_ids"] = media_ids
        if tagged_user_ids:
            self.params["media_tagged_user_ids"] = tagged_user_ids
        return self

    def add_deep_link(self, link: str):
        if link:
            self.params["direct_message_deep_link"] = link
        return self

    def add_super_followers(self, for_super_followers: bool):
        if for_super_followers:
            self.params["for_super_followers_only"] = for_super_followers
        return self

    def add_place(self, place_id: str):
        if place_id:
            self.params["place_id"] = place_id
        return self

    def add_poll_options(self, poll_options: list):
        if poll_options:
            self.params["poll_options"] = poll_options
        return self

    def add_poll_duration(self, poll_duration_minutes: int):
        if poll_duration_minutes:
            self.params["poll_duration_minutes"] = poll_duration_minutes
        return self

    def add_quote(self, quote_id: str):
        if quote_id:
            self.params["quote_tweet_id"] = quote_id
        return self

    def add_reply_settings(
        self,
        exclude_user_ids: list,
        reply_to_id: str,
        settings: TweetReplySettingsFilter,
    ):
        if exclude_user_ids:
            self.params["exclude_reply_user_ids"] = exclude_user_ids
        if reply_to_id:
            self.params["in_reply_to_tweet_id"] = reply_to_id
        if settings.All_Users:
            self.params["reply_settings"] = None
        elif settings.Following_Users_Only:
            self.params["reply_settings"] = "following"
        elif settings.Mentioned_Users_Only:
            self.params["reply_settings"] = "mentionedUsers"
        return self

    def build(self):
        return self.params


def _add_selected(
    params: Dict[str, Any],
    key: str,
    flags: BaseModel | None,
    to_backend: Callable[[str], str] = str,
) -> None:
    """Set ``key`` to the comma-joined backend names of the flags set to True."""
    if not flags:
        return
    selected = [name for name, value in flags.model_dump().items() if value is True]
    if selected:
        params[key] = ",".join(to_backend(name) for name in selected)
