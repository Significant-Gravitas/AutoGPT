"""What each expansion builder method adds to a Twitter API query."""

import pytest

from backend.blocks.twitter._builders import (
    DMExpansionsBuilder,
    ListExpansionsBuilder,
    SpaceExpansionsBuilder,
    TweetExpansionsBuilder,
    UserExpansionsBuilder,
)
from backend.blocks.twitter._types import (
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
    TweetUserFieldsFilter,
    UserExpansionsFilter,
)

# builder: (method, a filter with one flag set, the query param it adds)
_ADDED = {
    TweetExpansionsBuilder: [
        (
            "add_expansions",
            ExpansionFilter(Poll_IDs=True),
            "expansions",
            "attachments.poll_ids",
        ),
        (
            "add_media_fields",
            TweetMediaFieldsFilter(Width=True),
            "media.fields",
            "width",
        ),
        (
            "add_place_fields",
            TweetPlaceFieldsFilter(Country=True),
            "place.fields",
            "country",
        ),
        ("add_poll_fields", TweetPollFieldsFilter(Poll_ID=True), "poll.fields", "id"),
        ("add_tweet_fields", TweetFieldsFilter(Tweet_ID=True), "tweet.fields", "id"),
        ("add_user_fields", TweetUserFieldsFilter(User_ID=True), "user.fields", "id"),
    ],
    UserExpansionsBuilder: [
        (
            "add_expansions",
            UserExpansionsFilter(pinned_tweet_id=True),
            "expansions",
            "pinned_tweet_id",
        ),
        ("add_tweet_fields", TweetFieldsFilter(Tweet_ID=True), "tweet.fields", "id"),
        ("add_user_fields", TweetUserFieldsFilter(User_ID=True), "user.fields", "id"),
    ],
    ListExpansionsBuilder: [
        (
            "add_expansions",
            ListExpansionsFilter(List_Owner_ID=True),
            "expansions",
            "owner_id",
        ),
        ("add_list_fields", ListFieldsFilter(List_ID=True), "list.fields", "id"),
        ("add_user_fields", TweetUserFieldsFilter(User_ID=True), "user.fields", "id"),
    ],
    SpaceExpansionsBuilder: [
        ("add_expansions", SpaceExpansionsFilter(Hosts=True), "expansions", "host_ids"),
        ("add_space_fields", SpaceFieldsFilter(Space_ID=True), "space.fields", "id"),
        ("add_user_fields", TweetUserFieldsFilter(User_ID=True), "user.fields", "id"),
    ],
    DMExpansionsBuilder: [
        (
            "add_expansions",
            DMEventExpansionFilter(sender_id=True),
            "expansions",
            "sender_id",
        ),
        (
            "add_event_types",
            DMEventTypeFilter(MessageCreate=True),
            "event_types",
            "MessageCreate",
        ),
        ("add_media_fields", DMMediaFieldFilter(url=True), "media.fields", "url"),
        ("add_tweet_fields", DMTweetFieldFilter(id=True), "tweet.fields", "id"),
        # Unlike the other builders, DM user fields go out under the filter's own names.
        (
            "add_user_fields",
            TweetUserFieldsFilter(User_ID=True),
            "user.fields",
            "User_ID",
        ),
    ],
}
_CASES = [(builder, *case) for builder, cases in _ADDED.items() for case in cases]


@pytest.mark.parametrize("builder, method, flags, key, value", _CASES)
def test_a_set_flag_adds_its_query_param(builder, method, flags, key, value):
    assert getattr(builder({}), method)(flags).build() == {key: value}


@pytest.mark.parametrize("builder, method, flags, key, value", _CASES)
def test_no_set_flag_adds_nothing(builder, method, flags, key, value):
    assert getattr(builder({}), method)(None).build() == {}
    assert getattr(builder({}), method)(type(flags)()).build() == {}


def test_set_flags_are_joined_in_field_order():
    flags = TweetUserFieldsFilter(User_Bio=True, User_ID=True)
    built = TweetExpansionsBuilder({}).add_user_fields(flags).build()
    assert built == {"user.fields": "description,id"}
