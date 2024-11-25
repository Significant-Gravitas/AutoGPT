from typing import Any, Dict

from backend.blocks.twitter._types import TweetExpansions, TweetReplySettings
from backend.blocks.twitter._mappers import get_backend_expansion, get_backend_field, get_backend_list_expansion, get_backend_list_field, get_backend_media_field, get_backend_place_field, get_backend_poll_field, get_backend_space_expansion, get_backend_space_field, get_backend_user_field

# Common Builder
class TweetExpansionsBuilder:
    def __init__(self, param : Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: list[TweetExpansions]):
        if expansions:
            self.params["expansions"] = ",".join([get_backend_expansion(exp.name) for exp in expansions])
        return self

    def add_media_fields(self, media_fields: list):
        if media_fields:
            self.params["media.fields"] = ",".join([get_backend_media_field(field.name) for field in media_fields])
        return self

    def add_place_fields(self, place_fields: list):
        if place_fields:
            self.params["place.fields"] = ",".join([get_backend_place_field(field.name) for field in place_fields])
        return self

    def add_poll_fields(self, poll_fields: list):
        if poll_fields:
            self.params["poll.fields"] = ",".join([get_backend_poll_field(field.name) for field in poll_fields])
        return self

    def add_tweet_fields(self, tweet_fields: list):
        if tweet_fields:
            self.params["tweet.fields"] = ",".join([get_backend_field(field.name) for field in tweet_fields])
        return self

    def add_user_fields(self, user_fields: list):
        if user_fields:
            self.params["user.fields"] = ",".join([get_backend_user_field(field.name) for field in user_fields])
        return self

    def build(self):
        return self.params

class UserExpansionsBuilder:
    def __init__(self, param : Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: list):
        if expansions:
            self.params["expansions"] = ",".join([exp.value for exp in expansions])
        return self

    def add_tweet_fields(self, tweet_fields: list):
        if tweet_fields:
            self.params["tweet.fields"] = ",".join([get_backend_field(field.name) for field in tweet_fields])
        return self

    def add_user_fields(self, user_fields: list):
        if user_fields:
            self.params["user.fields"] = ",".join([get_backend_user_field(field.name) for field in user_fields])
        return self

    def build(self):
        return self.params

class ListExpansionsBuilder:
    def __init__(self, param : Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: list):
        if expansions:
            self.params["expansions"] = ",".join([get_backend_list_expansion(exp.name) for exp in expansions])
        return self

    def add_list_fields(self, list_fields: list):
        if list_fields:
            self.params["list.fields"] = ",".join([get_backend_list_field(field.name) for field in list_fields])
        return self

    def add_user_fields(self, user_fields: list):
        if user_fields:
            self.params["user.fields"] = ",".join([get_backend_user_field(field.name) for field in user_fields])
        return self

    def build(self):
        return self.params

class SpaceExpansionsBuilder:
    def __init__(self, param : Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: list):
        if expansions:
            self.params["expansions"] = ",".join([get_backend_space_expansion(exp.name) for exp in expansions])
        return self

    def add_space_fields(self, space_fields: list):
        if space_fields:
            self.params["space.fields"] = ",".join([get_backend_space_field(field.name) for field in space_fields])
        return self

    def add_user_fields(self, user_fields: list):
        if user_fields:
            self.params["user.fields"] = ",".join([get_backend_user_field(field.name) for field in user_fields])
        return self

    def build(self):
        return self.params

class TweetDurationBuilder:
    def __init__(self, param : Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_start_time(self, start_time: str):
        if start_time:
            self.params["start_time"] = start_time
        return self

    def add_end_time(self, end_time: str):
        if end_time:
            self.params["end_time"] = end_time
        return self

    def add_since_id(self, since_id: str):
        if since_id:
            self.params["since_id"] = since_id
        return self

    def add_until_id(self, until_id: str):
        if until_id:
            self.params["until_id"] = until_id
        return self

    def add_sort_order(self, sort_order: str):
        if sort_order:
            self.params["sort_order"] = sort_order
        return self

    def build(self):
        return self.params

class DMExpansionsBuilder:
    def __init__(self, param : Dict[str, Any]):
        self.params: Dict[str, Any] = param

    def add_expansions(self, expansions: list):
        if expansions:
            self.params["expansions"] = ",".join([exp.value for exp in expansions])
        return self

    def add_event_types(self, event_types: list):
        if event_types:
            self.params["event_types"] = ",".join([field.value for field in event_types])
        return self

    def add_media_fields(self, media_fields: list):
        if media_fields:
            self.params["media.fields"] = ",".join([field.value for field in media_fields])
        return self

    def add_tweet_fields(self, tweet_fields: list):
        if tweet_fields:
            self.params["tweet.fields"] = ",".join([field.value for field in tweet_fields])
        return self

    def add_user_fields(self, user_fields: list):
        if user_fields:
            self.params["user.fields"] = ",".join([field.value for field in user_fields])
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

    def add_pagination(self, max_results: int, pagination: str):
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

    def add_text(self, text: str):
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

    def add_poll_duration(self, poll_duration_minutes: int ):
        if poll_duration_minutes:
            self.params["poll_duration_minutes"] = poll_duration_minutes
        return self

    def add_quote(self, quote_id: str):
        if quote_id:
            self.params["quote_tweet_id"] = quote_id
        return self

    def add_reply_settings(self, exclude_user_ids: list, reply_to_id: str, settings: TweetReplySettings):
        if exclude_user_ids:
            self.params["exclude_reply_user_ids"] = exclude_user_ids
        if reply_to_id:
            self.params["in_reply_to_tweet_id"] = reply_to_id
        if settings == TweetReplySettings.all_users:
            self.params["reply_settings"] = None
        else:
            self.params["reply_settings"] = settings
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
