from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from backend.data.block import BlockSchema
from backend.data.model import SchemaField

# -------------- Tweets -----------------


class TweetReplySettingsFilter(BaseModel):
    Mentioned_Users_Only: bool = False
    Following_Users_Only: bool = False
    All_Users: bool = False


class TweetUserFieldsFilter(BaseModel):
    Account_Creation_Date: bool = False
    User_Bio: bool = False
    User_Entities: bool = False
    User_ID: bool = False
    User_Location: bool = False
    Latest_Tweet_ID: bool = False
    Display_Name: bool = False
    Pinned_Tweet_ID: bool = False
    Profile_Picture_URL: bool = False
    Is_Protected_Account: bool = False
    Account_Statistics: bool = False
    Profile_URL: bool = False
    Username: bool = False
    Is_Verified: bool = False
    Verification_Type: bool = False
    Content_Withholding_Info: bool = False


class TweetFieldsFilter(BaseModel):
    Tweet_Attachments: bool = False
    Author_ID: bool = False
    Context_Annotations: bool = False
    Conversation_ID: bool = False
    Creation_Time: bool = False
    Edit_Controls: bool = False
    Tweet_Entities: bool = False
    Geographic_Location: bool = False
    Tweet_ID: bool = False
    Reply_To_User_ID: bool = False
    Language: bool = False
    Public_Metrics: bool = False
    Sensitive_Content_Flag: bool = False
    Referenced_Tweets: bool = False
    Reply_Settings: bool = False
    Tweet_Source: bool = False
    Tweet_Text: bool = False
    Withheld_Content: bool = False


class PersonalTweetFieldsFilter(BaseModel):
    attachments: bool = False
    author_id: bool = False
    context_annotations: bool = False
    conversation_id: bool = False
    created_at: bool = False
    edit_controls: bool = False
    entities: bool = False
    geo: bool = False
    id: bool = False
    in_reply_to_user_id: bool = False
    lang: bool = False
    non_public_metrics: bool = False
    public_metrics: bool = False
    organic_metrics: bool = False
    promoted_metrics: bool = False
    possibly_sensitive: bool = False
    referenced_tweets: bool = False
    reply_settings: bool = False
    source: bool = False
    text: bool = False
    withheld: bool = False


class TweetPollFieldsFilter(BaseModel):
    Duration_Minutes: bool = False
    End_DateTime: bool = False
    Poll_ID: bool = False
    Poll_Options: bool = False
    Voting_Status: bool = False


class TweetPlaceFieldsFilter(BaseModel):
    Contained_Within_Places: bool = False
    Country: bool = False
    Country_Code: bool = False
    Full_Location_Name: bool = False
    Geographic_Coordinates: bool = False
    Place_ID: bool = False
    Place_Name: bool = False
    Place_Type: bool = False


class TweetMediaFieldsFilter(BaseModel):
    Duration_in_Milliseconds: bool = False
    Height: bool = False
    Media_Key: bool = False
    Preview_Image_URL: bool = False
    Media_Type: bool = False
    Media_URL: bool = False
    Width: bool = False
    Public_Metrics: bool = False
    Non_Public_Metrics: bool = False
    Organic_Metrics: bool = False
    Promoted_Metrics: bool = False
    Alternative_Text: bool = False
    Media_Variants: bool = False


class ExpansionFilter(BaseModel):
    Poll_IDs: bool = False
    Media_Keys: bool = False
    Author_User_ID: bool = False
    Edit_History_Tweet_IDs: bool = False
    Mentioned_Usernames: bool = False
    Place_ID: bool = False
    Reply_To_User_ID: bool = False
    Referenced_Tweet_ID: bool = False
    Referenced_Tweet_Author_ID: bool = False


class TweetExcludesFilter(BaseModel):
    retweets: bool = False
    replies: bool = False


# -------------- Users -----------------


class UserExpansionsFilter(BaseModel):
    pinned_tweet_id: bool = False


# -------------- DM's' -----------------


class DMEventFieldFilter(BaseModel):
    id: bool = False
    text: bool = False
    event_type: bool = False
    created_at: bool = False
    dm_conversation_id: bool = False
    sender_id: bool = False
    participant_ids: bool = False
    referenced_tweets: bool = False
    attachments: bool = False


class DMEventTypeFilter(BaseModel):
    MessageCreate: bool = False
    ParticipantsJoin: bool = False
    ParticipantsLeave: bool = False


class DMEventExpansionFilter(BaseModel):
    attachments_media_keys: bool = False
    referenced_tweets_id: bool = False
    sender_id: bool = False
    participant_ids: bool = False


class DMMediaFieldFilter(BaseModel):
    duration_ms: bool = False
    height: bool = False
    media_key: bool = False
    preview_image_url: bool = False
    type: bool = False
    url: bool = False
    width: bool = False
    public_metrics: bool = False
    alt_text: bool = False
    variants: bool = False


class DMTweetFieldFilter(BaseModel):
    attachments: bool = False
    author_id: bool = False
    context_annotations: bool = False
    conversation_id: bool = False
    created_at: bool = False
    edit_controls: bool = False
    entities: bool = False
    geo: bool = False
    id: bool = False
    in_reply_to_user_id: bool = False
    lang: bool = False
    public_metrics: bool = False
    possibly_sensitive: bool = False
    referenced_tweets: bool = False
    reply_settings: bool = False
    source: bool = False
    text: bool = False
    withheld: bool = False


# -------------- Spaces -----------------


class SpaceExpansionsFilter(BaseModel):
    Invited_Users: bool = False
    Speakers: bool = False
    Creator: bool = False
    Hosts: bool = False
    Topics: bool = False


class SpaceFieldsFilter(BaseModel):
    Space_ID: bool = False
    Space_State: bool = False
    Creation_Time: bool = False
    End_Time: bool = False
    Host_User_IDs: bool = False
    Language: bool = False
    Is_Ticketed: bool = False
    Invited_User_IDs: bool = False
    Participant_Count: bool = False
    Subscriber_Count: bool = False
    Scheduled_Start_Time: bool = False
    Speaker_User_IDs: bool = False
    Start_Time: bool = False
    Space_Title: bool = False
    Topic_IDs: bool = False
    Last_Updated_Time: bool = False


class SpaceStatesFilter(str, Enum):
    live = "live"
    scheduled = "scheduled"
    all = "all"


# -------------- List Expansions -----------------


class ListExpansionsFilter(BaseModel):
    List_Owner_ID: bool = False


class ListFieldsFilter(BaseModel):
    List_ID: bool = False
    List_Name: bool = False
    Creation_Date: bool = False
    Description: bool = False
    Follower_Count: bool = False
    Member_Count: bool = False
    Is_Private: bool = False
    Owner_ID: bool = False


# ---------  [Input Types] -------------
class TweetExpansionInputs(BlockSchema):

    expansions: ExpansionFilter | None = SchemaField(
        description="Choose what extra information you want to get with your tweets. For example:\n- Select 'Media_Keys' to get media details\n- Select 'Author_User_ID' to get user information\n- Select 'Place_ID' to get location details",
        placeholder="Pick the extra information you want to see",
        default=None,
        advanced=True,
    )

    media_fields: TweetMediaFieldsFilter | None = SchemaField(
        description="Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above.",
        placeholder="Choose what media details you want to see",
        default=None,
        advanced=True,
    )

    place_fields: TweetPlaceFieldsFilter | None = SchemaField(
        description="Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above.",
        placeholder="Choose what location details you want to see",
        default=None,
        advanced=True,
    )

    poll_fields: TweetPollFieldsFilter | None = SchemaField(
        description="Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above.",
        placeholder="Choose what poll details you want to see",
        default=None,
        advanced=True,
    )

    tweet_fields: TweetFieldsFilter | None = SchemaField(
        description="Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above.",
        placeholder="Choose what tweet details you want to see",
        default=None,
        advanced=True,
    )

    user_fields: TweetUserFieldsFilter | None = SchemaField(
        description="Select what user information you want to see. To use this, you must first select one of these in expansions above:\n- 'Author_User_ID' for tweet authors\n- 'Mentioned_Usernames' for mentioned users\n- 'Reply_To_User_ID' for users being replied to\n- 'Referenced_Tweet_Author_ID' for authors of referenced tweets",
        placeholder="Choose what user details you want to see",
        default=None,
        advanced=True,
    )


class DMEventExpansionInputs(BlockSchema):
    expansions: DMEventExpansionFilter | None = SchemaField(
        description="Select expansions to include related data objects in the 'includes' section.",
        placeholder="Enter expansions",
        default=None,
        advanced=True,
    )

    event_types: DMEventTypeFilter | None = SchemaField(
        description="Select DM event types to include in the response.",
        placeholder="Enter event types",
        default=None,
        advanced=True,
    )

    media_fields: DMMediaFieldFilter | None = SchemaField(
        description="Select media fields to include in the response (requires expansions=attachments.media_keys).",
        placeholder="Enter media fields",
        default=None,
        advanced=True,
    )

    tweet_fields: DMTweetFieldFilter | None = SchemaField(
        description="Select tweet fields to include in the response (requires expansions=referenced_tweets.id).",
        placeholder="Enter tweet fields",
        default=None,
        advanced=True,
    )

    user_fields: TweetUserFieldsFilter | None = SchemaField(
        description="Select user fields to include in the response (requires expansions=sender_id or participant_ids).",
        placeholder="Enter user fields",
        default=None,
        advanced=True,
    )


class UserExpansionInputs(BlockSchema):
    expansions: UserExpansionsFilter | None = SchemaField(
        description="Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet.",
        placeholder="Select extra user information to include",
        default=None,
        advanced=True,
    )

    tweet_fields: TweetFieldsFilter | None = SchemaField(
        description="Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above.",
        placeholder="Choose what details to see in pinned tweets",
        default=None,
        advanced=True,
    )

    user_fields: TweetUserFieldsFilter | None = SchemaField(
        description="Select what user information you want to see, like username, bio, profile picture, etc.",
        placeholder="Choose what user details you want to see",
        default=None,
        advanced=True,
    )


class SpaceExpansionInputs(BlockSchema):
    expansions: SpaceExpansionsFilter | None = SchemaField(
        description="Choose additional information you want to get with your Twitter Spaces:\n- Select 'Invited_Users' to see who was invited\n- Select 'Speakers' to see who can speak\n- Select 'Creator' to get details about who made the Space\n- Select 'Hosts' to see who's hosting\n- Select 'Topics' to see Space topics",
        placeholder="Pick what extra information you want to see about the Space",
        default=None,
        advanced=True,
    )

    space_fields: SpaceFieldsFilter | None = SchemaField(
        description="Choose what Space details you want to see, such as:\n- Title\n- Start/End times\n- Number of participants\n- Language\n- State (live/scheduled)\n- And more",
        placeholder="Choose what Space information you want to get",
        default=SpaceFieldsFilter(Space_Title=True, Host_User_IDs=True),
        advanced=True,
    )

    user_fields: TweetUserFieldsFilter | None = SchemaField(
        description="Choose what user information you want to see. This works when you select any of these in expansions above:\n- 'Creator' for Space creator details\n- 'Hosts' for host information\n- 'Speakers' for speaker details\n- 'Invited_Users' for invited user information",
        placeholder="Pick what details you want to see about the users",
        default=None,
        advanced=True,
    )


class ListExpansionInputs(BlockSchema):
    expansions: ListExpansionsFilter | None = SchemaField(
        description="Choose what extra information you want to get with your Twitter Lists:\n- Select 'List_Owner_ID' to get details about who owns the list\n\nThis will let you see more details about the list owner when you also select user fields below.",
        placeholder="Pick what extra list information you want to see",
        default=ListExpansionsFilter(List_Owner_ID=True),
        advanced=True,
    )

    user_fields: TweetUserFieldsFilter | None = SchemaField(
        description="Choose what information you want to see about list owners. This only works when you select 'List_Owner_ID' in expansions above.\n\nYou can see things like:\n- Their username\n- Profile picture\n- Account details\n- And more",
        placeholder="Select what details you want to see about list owners",
        default=TweetUserFieldsFilter(User_ID=True, Username=True),
        advanced=True,
    )

    list_fields: ListFieldsFilter | None = SchemaField(
        description="Choose what information you want to see about the Twitter Lists themselves, such as:\n- List name\n- Description\n- Number of followers\n- Number of members\n- Whether it's private\n- Creation date\n- And more",
        placeholder="Pick what list details you want to see",
        default=ListFieldsFilter(Owner_ID=True),
        advanced=True,
    )


class TweetTimeWindowInputs(BlockSchema):
    start_time: datetime | None = SchemaField(
        description="Start time in YYYY-MM-DDTHH:mm:ssZ format",
        placeholder="Enter start time",
        default=None,
        advanced=False,
    )

    end_time: datetime | None = SchemaField(
        description="End time in YYYY-MM-DDTHH:mm:ssZ format",
        placeholder="Enter end time",
        default=None,
        advanced=False,
    )

    since_id: str | None = SchemaField(
        description="Returns results with Tweet ID  greater than this (more recent than), we give priority to since_id over start_time",
        placeholder="Enter since ID",
        default=None,
        advanced=True,
    )

    until_id: str | None = SchemaField(
        description="Returns results with Tweet ID less than this (that is, older than), and used with since_id",
        placeholder="Enter until ID",
        default=None,
        advanced=True,
    )

    sort_order: str | None = SchemaField(
        description="Order of returned tweets (recency or relevancy)",
        placeholder="Enter sort order",
        default=None,
        advanced=True,
    )
