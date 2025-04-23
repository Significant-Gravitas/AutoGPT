from datetime import datetime
from typing import Any, Dict

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
        if expansions:
            filtered_expansions = [
                name for name, value in expansions.dict().items() if value is True
            ]

            if filtered_expansions:
                self.params["expansions"] = ",".join(
                    [get_backend_expansion(exp) for exp in filtered_expansions]
                )

        return self

    def add_media_fields(self, media_fields: TweetMediaFieldsFilter | None):
        if media_fields:
            filtered_fields = [
                name for name, value in media_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["media.fields"] = ",".join(
                    [get_backend_media_field(field) for field in filtered_fields]
                )
        return self

    def add_place_fields(self, place_fields: TweetPlaceFieldsFilter | None):
        if place_fields:
            filtered_fields = [
                name for name, value in place_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["place.fields"] = ",".join(
                    [get_backend_place_field(field) for field in filtered_fields]
                )
        return self

    def add_poll_fields(self, poll_fields: TweetPollFieldsFilter | None):
        if poll_fields:
            filtered_fields = [
                name for name, value in poll_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["poll.fields"] = ",".join(
                    [get_backend_poll_field(field) for field in filtered_fields]
                )
        return self

    def add_tweet_fields(self, tweet_fields: TweetFieldsFilter | None):
        if tweet_fields:
            filtered_fields = [
                name for name, value in tweet_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["tweet.fields"] = ",".join(
                    [get_backend_field(field) for field in filtered_fields]
                )
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        if user_fields:
            filtered_fields = [
                name for name, value in user_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["user.fields"] = ",".join(
                    [get_backend_user_field(field) for field in filtered_fields]
                )
        return self

    def build(self):
        return self.params


class UserExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: UserExpansionsFilter | None):
        if expansions:
            filtered_expansions = [
                name for name, value in expansions.dict().items() if value is True
            ]
            if filtered_expansions:
                self.params["expansions"] = ",".join(filtered_expansions)
        return self

    def add_tweet_fields(self, tweet_fields: TweetFieldsFilter | None):
        if tweet_fields:
            filtered_fields = [
                name for name, value in tweet_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["tweet.fields"] = ",".join(
                    [get_backend_field(field) for field in filtered_fields]
                )
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        if user_fields:
            filtered_fields = [
                name for name, value in user_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["user.fields"] = ",".join(
                    [get_backend_user_field(field) for field in filtered_fields]
                )
        return self

    def build(self):
        return self.params


class ListExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: ListExpansionsFilter | None):
        if expansions:
            filtered_expansions = [
                name for name, value in expansions.dict().items() if value is True
            ]
            if filtered_expansions:
                self.params["expansions"] = ",".join(
                    [get_backend_list_expansion(exp) for exp in filtered_expansions]
                )
        return self

    def add_list_fields(self, list_fields: ListFieldsFilter | None):
        if list_fields:
            filtered_fields = [
                name for name, value in list_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["list.fields"] = ",".join(
                    [get_backend_list_field(field) for field in filtered_fields]
                )
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        if user_fields:
            filtered_fields = [
                name for name, value in user_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["user.fields"] = ",".join(
                    [get_backend_user_field(field) for field in filtered_fields]
                )
        return self

    def build(self):
        return self.params


class SpaceExpansionsBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: SpaceExpansionsFilter | None):
        if expansions:
            filtered_expansions = [
                name for name, value in expansions.dict().items() if value is True
            ]
            if filtered_expansions:
                self.params["expansions"] = ",".join(
                    [get_backend_space_expansion(exp) for exp in filtered_expansions]
                )
        return self

    def add_space_fields(self, space_fields: SpaceFieldsFilter | None):
        if space_fields:
            filtered_fields = [
                name for name, value in space_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["space.fields"] = ",".join(
                    [get_backend_space_field(field) for field in filtered_fields]
                )
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter | None):
        if user_fields:
            filtered_fields = [
                name for name, value in user_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["user.fields"] = ",".join(
                    [get_backend_user_field(field) for field in filtered_fields]
                )
        return self

    def build(self):
        return self.params


class TweetDurationBuilder:
    def __init__(self, param: Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_start_time(self, start_time: datetime | None):
        if start_time:
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
        if expansions:
            filtered_expansions = [
                name for name, value in expansions.dict().items() if value is True
            ]
            if filtered_expansions:
                self.params["expansions"] = ",".join(filtered_expansions)
        return self

    def add_event_types(self, event_types: DMEventTypeFilter):
        if event_types:
            filtered_types = [
                name for name, value in event_types.dict().items() if value is True
            ]
            if filtered_types:
                self.params["event_types"] = ",".join(filtered_types)
        return self

    def add_media_fields(self, media_fields: DMMediaFieldFilter):
        if media_fields:
            filtered_fields = [
                name for name, value in media_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["media.fields"] = ",".join(filtered_fields)
        return self

    def add_tweet_fields(self, tweet_fields: DMTweetFieldFilter):
        if tweet_fields:
            filtered_fields = [
                name for name, value in tweet_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["tweet.fields"] = ",".join(filtered_fields)
        return self

    def add_user_fields(self, user_fields: TweetUserFieldsFilter):
        if user_fields:
            filtered_fields = [
                name for name, value in user_fields.dict().items() if value is True
            ]
            if filtered_fields:
                self.params["user.fields"] = ",".join(filtered_fields)
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


class TweetGetsBuilder:
    def __init__(self):
        self.params: Dict[str, Any] = {"user_auth": False}

    def add_id(self, tweet_id: list[str]):
        self.params["id"] = tweet_id
        return self

    def build(self):
        return self.params
