from enum import Enum

from backend.data.block import BlockSchema
from backend.data.model import SchemaField

# -------------- Tweets -----------------


class TweetReplySettings(str, Enum):
    mentioned_users = "Mentioned_Users_Only"
    following = "Following_Users_Only"
    all_users = "All_Users"


class TweetUserFields(str, Enum):
    created_at = "Account_Creation_Date"
    description = "User_Bio"
    entities = "User_Entities"
    id = "User_ID"
    location = "User_Location"
    most_recent_tweet_id = "Latest_Tweet_ID"
    name_user = "Display_Name"
    pinned_tweet_id = "Pinned_Tweet_ID"
    profile_image_url = "Profile_Picture_URL"
    protected = "Is_Protected_Account"
    public_metrics = "Account_Statistics"
    url = "Profile_URL"
    username = "Username"
    verified = "Is_Verified"
    verified_type = "Verification_Type"
    withheld = "Content_Withholding_Info"


class TweetFields(str, Enum):
    attachments = "Tweet_Attachments"
    author_id = "Author_ID"
    context_annotations = "Context_Annotations"
    conversation_id = "Conversation_ID"
    created_at = "Creation_Time"
    edit_controls = "Edit_Controls"
    entities = "Tweet_Entities"
    geo = "Geographic_Location"
    id = "Tweet_ID"
    in_reply_to_user_id = "Reply_To_User_ID"
    lang = "Language"
    public_metrics = "Public_Metrics"
    possibly_sensitive = "Sensitive_Content_Flag"
    referenced_tweets = "Referenced_Tweets"
    reply_settings = "Reply_Settings"
    source = "Tweet_Source"
    text = "Tweet_Text"
    withheld = "Withheld_Content"


class PersonalTweetFields(str, Enum):
    attachments = "attachments"
    author_id = "author_id"
    context_annotations = "context_annotations"
    conversation_id = "conversation_id"
    created_at = "created_at"
    edit_controls = "edit_controls"
    entities = "entities"
    geo = "geo"
    id = "id"
    in_reply_to_user_id = "in_reply_to_user_id"
    lang = "lang"
    non_public_metrics = "non_public_metrics"
    public_metrics = "public_metrics"
    organic_metrics = "organic_metrics"
    promoted_metrics = "promoted_metrics"
    possibly_sensitive = "possibly_sensitive"
    referenced_tweets = "referenced_tweets"
    reply_settings = "reply_settings"
    source = "source"
    text = "text"
    withheld = "withheld"


class TweetPollFields(str, Enum):
    duration_minutes = "Duration_Minutes"
    end_datetime = "End_DateTime"
    id = "Poll_ID"
    options = "Poll_Options"
    voting_status = "Voting_Status"


class TweetPlaceFields(str, Enum):
    contained_within = "Contained_Within_Places"
    country = "Country"
    country_code = "Country_Code"
    full_name = "Full_Location_Name"
    geo = "Geographic_Coordinates"
    id = "Place_ID"
    place_name = "Place_Name"
    place_type = "Place_Type"


class TweetMediaFields(str, Enum):
    duration_ms = "Duration_in_Milliseconds"
    height = "Height"
    media_key = "Media_Key"
    preview_image_url = "Preview_Image_URL"
    type = "Media_Type"
    url = "Media_URL"
    width = "Width"
    public_metrics = "Public_Metrics"
    non_public_metrics = "Non_Public_Metrics"
    organic_metrics = "Organic_Metrics"
    promoted_metrics = "Promoted_Metrics"
    alt_text = "Alternative_Text"
    variants = "Media_Variants"


class TweetExpansions(str, Enum):
    attachments_poll_ids = "Poll_IDs"
    attachments_media_keys = "Media_Keys"
    author_id = "Author_User_ID"
    edit_history_tweet_ids = "Edit_History_Tweet_IDs"
    entities_mentions_username = "Mentioned_Usernames"
    geo_place_id = "Place_ID"
    in_reply_to_user_id = "Reply_To_User_ID"
    referenced_tweets_id = "Referenced_Tweet_ID"
    referenced_tweets_id_author_id = "Referenced_Tweet_Author_ID"


class TweetExcludes(str, Enum):
    retweets = "retweets"
    replies = "replies"


# -------------- Users -----------------


class UserExpansions(str, Enum):
    pinned_tweet_id = "pinned_tweet_id"


# -------------- DM's' -----------------


class DMEventField(str, Enum):
    ID = "id"
    TEXT = "text"
    EVENT_TYPE = "event_type"
    CREATED_AT = "created_at"
    DM_CONVERSATION_ID = "dm_conversation_id"
    SENDER_ID = "sender_id"
    PARTICIPANT_IDS = "participant_ids"
    REFERENCED_TWEETS = "referenced_tweets"
    ATTACHMENTS = "attachments"


class DMEventType(str, Enum):
    MESSAGE_CREATE = "MessageCreate"
    PARTICIPANTS_JOIN = "ParticipantsJoin"
    PARTICIPANTS_LEAVE = "ParticipantsLeave"


class DMEventExpansion(str, Enum):
    ATTACHMENTS_MEDIA_KEYS = "attachments.media_keys"
    REFERENCED_TWEETS_ID = "referenced_tweets.id"
    SENDER_ID = "sender_id"
    PARTICIPANT_IDS = "participant_ids"


class DMMediaField(str, Enum):
    DURATION_MS = "duration_ms"
    HEIGHT = "height"
    MEDIA_KEY = "media_key"
    PREVIEW_IMAGE_URL = "preview_image_url"
    TYPE = "type"
    URL = "url"
    WIDTH = "width"
    PUBLIC_METRICS = "public_metrics"
    ALT_TEXT = "alt_text"
    VARIANTS = "variants"


class DMTweetField(str, Enum):
    ATTACHMENTS = "attachments"
    AUTHOR_ID = "author_id"
    CONTEXT_ANNOTATIONS = "context_annotations"
    CONVERSATION_ID = "conversation_id"
    CREATED_AT = "created_at"
    EDIT_CONTROLS = "edit_controls"
    ENTITIES = "entities"
    GEO = "geo"
    ID = "id"
    IN_REPLY_TO_USER_ID = "in_reply_to_user_id"
    LANG = "lang"
    PUBLIC_METRICS = "public_metrics"
    POSSIBLY_SENSITIVE = "possibly_sensitive"
    REFERENCED_TWEETS = "referenced_tweets"
    REPLY_SETTINGS = "reply_settings"
    SOURCE = "source"
    TEXT = "text"
    WITHHELD = "withheld"


# -------------- Spaces -----------------


class SpaceExpansions(str, Enum):
    invited_user_ids = "Invited_Users"
    speaker_ids = "Speakers"
    creator_id = "Creator"
    host_ids = "Hosts"
    topic_ids = "Topics"


class SpaceFields(str, Enum):
    id = "Space_ID"
    state = "Space_State"
    created_at = "Creation_Time"
    ended_at = "End_Time"
    host_ids = "Host_User_IDs"
    lang = "Language"
    is_ticketed = "Is_Ticketed"
    invited_user_ids = "Invited_User_IDs"
    participant_count = "Participant_Count"
    subscriber_count = "Subscriber_Count"
    scheduled_start = "Scheduled_Start_Time"
    speaker_ids = "Speaker_User_IDs"
    started_at = "Start_Time"
    title_ = "Space_Title"
    topic_ids = "Topic_IDs"
    updated_at = "Last_Updated_Time"


class SpaceStates(str, Enum):
    LIVE = "live"
    SCHEDULED = "scheduled"
    ALL = "all"


# -------------- List Expansions -----------------


class ListExpansions(str, Enum):
    owner_id = "List_Owner_ID"


class ListFields(str, Enum):
    id = "List_ID"
    list_name = "List_Name"
    created_at = "Creation_Date"
    description = "Description"
    follower_count = "Follower_Count"
    member_count = "Member_Count"
    private = "Is_Private"
    owner_id = "Owner_ID"


# ---------  [Input Types] -------------
class TweetExpansionInputs(BlockSchema):
    expansions: list[TweetExpansions] = SchemaField(
        description="Choose what extra information you want to get with your tweets. For example:\n- Select 'Media_Keys' to get media details\n- Select 'Author_User_ID' to get user information\n- Select 'Place_ID' to get location details",
        enum=TweetExpansions,
        placeholder="Pick the extra information you want to see",
        default=[],
        is_multi_select=True,
        advanced=True,
    )

    media_fields: list[TweetMediaFields] = SchemaField(
        description="Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above.",
        enum=TweetMediaFields,
        placeholder="Choose what media details you want to see",
        default=[],
        is_multi_select=True,
        advanced=True,
    )

    place_fields: list[TweetPlaceFields] = SchemaField(
        description="Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above.",
        placeholder="Choose what location details you want to see",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetPlaceFields,
    )

    poll_fields: list[TweetPollFields] = SchemaField(
        description="Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above.",
        placeholder="Choose what poll details you want to see",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetPollFields,
    )

    tweet_fields: list[TweetFields] = SchemaField(
        description="Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above.",
        placeholder="Choose what tweet details you want to see",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetFields,
    )

    user_fields: list[TweetUserFields] = SchemaField(
        description="Select what user information you want to see. To use this, you must first select one of these in expansions above:\n- 'Author_User_ID' for tweet authors\n- 'Mentioned_Usernames' for mentioned users\n- 'Reply_To_User_ID' for users being replied to\n- 'Referenced_Tweet_Author_ID' for authors of referenced tweets",
        placeholder="Choose what user details you want to see",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetUserFields,
    )


class DMEventExpansionInputs(BlockSchema):
    expansions: list[DMEventExpansion] = SchemaField(
        description="Select expansions to include related data objects in the 'includes' section.",
        enum=DMEventExpansion,
        placeholder="Enter expansions",
        default=[],
        is_multi_select=True,
        advanced=True,
    )

    event_types: list[DMEventType] = SchemaField(
        description="Select DM event types to include in the response.",
        placeholder="Enter event types",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=DMEventType,
    )

    media_fields: list[DMMediaField] = SchemaField(
        description="Select media fields to include in the response (requires expansions=attachments.media_keys).",
        placeholder="Enter media fields",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=DMMediaField,
    )

    tweet_fields: list[DMTweetField] = SchemaField(
        description="Select tweet fields to include in the response (requires expansions=referenced_tweets.id).",
        placeholder="Enter tweet fields",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=DMTweetField,
    )

    user_fields: list[TweetUserFields] = SchemaField(
        description="Select user fields to include in the response (requires expansions=sender_id or participant_ids).",
        placeholder="Enter user fields",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetUserFields,
    )


class UserExpansionInputs(BlockSchema):
    expansions: list[UserExpansions] = SchemaField(
        description="Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet.",
        enum=UserExpansions,
        placeholder="Select extra user information to include",
        default=[],
        is_multi_select=True,
        advanced=True,
    )

    tweet_fields: list[TweetFields] = SchemaField(
        description="Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above.",
        placeholder="Choose what details to see in pinned tweets",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetFields,
    )

    user_fields: list[TweetUserFields] = SchemaField(
        description="Select what user information you want to see, like username, bio, profile picture, etc.",
        placeholder="Choose what user details you want to see",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetUserFields,
    )


class SpaceExpansionInputs(BlockSchema):
    expansions: list[SpaceExpansions] = SchemaField(
        description="Choose additional information you want to get with your Twitter Spaces:\n- Select 'Invited_Users' to see who was invited\n- Select 'Speakers' to see who can speak\n- Select 'Creator' to get details about who made the Space\n- Select 'Hosts' to see who's hosting\n- Select 'Topics' to see Space topics",
        enum=SpaceExpansions,
        placeholder="Pick what extra information you want to see about the Space",
        default=[],
        is_multi_select=True,
        advanced=True,
    )

    space_fields: list[SpaceFields] = SchemaField(
        description="Choose what Space details you want to see, such as:\n- Title\n- Start/End times\n- Number of participants\n- Language\n- State (live/scheduled)\n- And more",
        placeholder="Choose what Space information you want to get",
        default=[SpaceFields.title_, SpaceFields.host_ids],
        advanced=True,
        is_multi_select=True,
        enum=SpaceFields,
    )

    user_fields: list[TweetUserFields] = SchemaField(
        description="Choose what user information you want to see. This works when you select any of these in expansions above:\n- 'Creator' for Space creator details\n- 'Hosts' for host information\n- 'Speakers' for speaker details\n- 'Invited_Users' for invited user information",
        placeholder="Pick what details you want to see about the users",
        default=[],
        advanced=True,
        is_multi_select=True,
        enum=TweetUserFields,
    )


class ListExpansionInputs(BlockSchema):
    expansions: list[ListExpansions] = SchemaField(
        description="Choose what extra information you want to get with your Twitter Lists:\n- Select 'List_Owner_ID' to get details about who owns the list\n\nThis will let you see more details about the list owner when you also select user fields below.",
        enum=ListExpansions,
        placeholder="Pick what extra list information you want to see",
        default=[ListExpansions.owner_id],
        is_multi_select=True,
        advanced=True,
    )

    user_fields: list[TweetUserFields] = SchemaField(
        description="Choose what information you want to see about list owners. This only works when you select 'List_Owner_ID' in expansions above.\n\nYou can see things like:\n- Their username\n- Profile picture\n- Account details\n- And more",
        placeholder="Select what details you want to see about list owners",
        default=[TweetUserFields.id, TweetUserFields.username],
        advanced=True,
        is_multi_select=True,
        enum=TweetUserFields,
    )

    list_fields: list[ListFields] = SchemaField(
        description="Choose what information you want to see about the Twitter Lists themselves, such as:\n- List name\n- Description\n- Number of followers\n- Number of members\n- Whether it's private\n- Creation date\n- And more",
        placeholder="Pick what list details you want to see",
        default=[ListFields.owner_id],
        advanced=True,
        is_multi_select=True,
        enum=ListFields,
    )


class TweetTimeWindowInputs(BlockSchema):
    start_time: str = SchemaField(
        description="Start time in YYYY-MM-DDTHH:mm:ssZ format",
        placeholder="Enter start time",
        default="",
    )

    end_time: str = SchemaField(
        description="End time in YYYY-MM-DDTHH:mm:ssZ format",
        default="",
        placeholder="Enter end time",
    )

    since_id: str = SchemaField(
        description="Returns results with Tweet ID  greater than this (more recent than), we give priority to since_id over start_time",
        default="",
        placeholder="Enter since ID",
    )

    until_id: str = SchemaField(
        description="Returns results with Tweet ID less than this (that is, older than), and used with since_id",
        default="",
        placeholder="Enter until ID",
    )

    sort_order: str = SchemaField(
        description="Order of returned tweets (recency or relevancy)",
        default="",
        placeholder="Enter sort order",
    )
