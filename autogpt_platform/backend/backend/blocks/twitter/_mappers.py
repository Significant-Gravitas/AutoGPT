# -------------- Tweets -----------------

# Tweet Expansions
EXPANSION_FRONTEND_TO_BACKEND_MAPPING = {
    "Poll_IDs": "attachments.poll_ids",
    "Media_Keys": "attachments.media_keys",
    "Author_User_ID": "author_id",
    "Edit_History_Tweet_IDs": "edit_history_tweet_ids",
    "Mentioned_Usernames": "entities.mentions.username",
    "Place_ID": "geo.place_id",
    "Reply_To_User_ID": "in_reply_to_user_id",
    "Referenced_Tweet_ID": "referenced_tweets.id",
    "Referenced_Tweet_Author_ID": "referenced_tweets.id.author_id",
}


def get_backend_expansion(frontend_key: str) -> str:
    result = EXPANSION_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid expansion key: {frontend_key}")
    return result


# TweetReplySettings
REPLY_SETTINGS_FRONTEND_TO_BACKEND_MAPPING = {
    "Mentioned_Users_Only": "mentionedUsers",
    "Following_Users_Only": "following",
    "All_Users": "all",
}


# TweetUserFields
def get_backend_reply_setting(frontend_key: str) -> str:
    result = REPLY_SETTINGS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid reply setting key: {frontend_key}")
    return result


USER_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "Account_Creation_Date": "created_at",
    "User_Bio": "description",
    "User_Entities": "entities",
    "User_ID": "id",
    "User_Location": "location",
    "Latest_Tweet_ID": "most_recent_tweet_id",
    "Display_Name": "name",
    "Pinned_Tweet_ID": "pinned_tweet_id",
    "Profile_Picture_URL": "profile_image_url",
    "Is_Protected_Account": "protected",
    "Account_Statistics": "public_metrics",
    "Profile_URL": "url",
    "Username": "username",
    "Is_Verified": "verified",
    "Verification_Type": "verified_type",
    "Content_Withholding_Info": "withheld",
}


def get_backend_user_field(frontend_key: str) -> str:
    result = USER_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid user field key: {frontend_key}")
    return result


# TweetFields
FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "Tweet_Attachments": "attachments",
    "Author_ID": "author_id",
    "Context_Annotations": "context_annotations",
    "Conversation_ID": "conversation_id",
    "Creation_Time": "created_at",
    "Edit_Controls": "edit_controls",
    "Tweet_Entities": "entities",
    "Geographic_Location": "geo",
    "Tweet_ID": "id",
    "Reply_To_User_ID": "in_reply_to_user_id",
    "Language": "lang",
    "Public_Metrics": "public_metrics",
    "Sensitive_Content_Flag": "possibly_sensitive",
    "Referenced_Tweets": "referenced_tweets",
    "Reply_Settings": "reply_settings",
    "Tweet_Source": "source",
    "Tweet_Text": "text",
    "Withheld_Content": "withheld",
}


def get_backend_field(frontend_key: str) -> str:
    result = FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid field key: {frontend_key}")
    return result


# TweetPollFields
POLL_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "Duration_Minutes": "duration_minutes",
    "End_DateTime": "end_datetime",
    "Poll_ID": "id",
    "Poll_Options": "options",
    "Voting_Status": "voting_status",
}


def get_backend_poll_field(frontend_key: str) -> str:
    result = POLL_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid poll field key: {frontend_key}")
    return result


PLACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "Contained_Within_Places": "contained_within",
    "Country": "country",
    "Country_Code": "country_code",
    "Full_Location_Name": "full_name",
    "Geographic_Coordinates": "geo",
    "Place_ID": "id",
    "Place_Name": "name",
    "Place_Type": "place_type",
}


def get_backend_place_field(frontend_key: str) -> str:
    result = PLACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid place field key: {frontend_key}")
    return result


# TweetMediaFields
MEDIA_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "Duration_in_Milliseconds": "duration_ms",
    "Height": "height",
    "Media_Key": "media_key",
    "Preview_Image_URL": "preview_image_url",
    "Media_Type": "type",
    "Media_URL": "url",
    "Width": "width",
    "Public_Metrics": "public_metrics",
    "Non_Public_Metrics": "non_public_metrics",
    "Organic_Metrics": "organic_metrics",
    "Promoted_Metrics": "promoted_metrics",
    "Alternative_Text": "alt_text",
    "Media_Variants": "variants",
}


def get_backend_media_field(frontend_key: str) -> str:
    result = MEDIA_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid media field key: {frontend_key}")
    return result


# -------------- Spaces -----------------

# SpaceExpansions
EXPANSION_FRONTEND_TO_BACKEND_MAPPING_SPACE = {
    "Invited_Users": "invited_user_ids",
    "Speakers": "speaker_ids",
    "Creator": "creator_id",
    "Hosts": "host_ids",
    "Topics": "topic_ids",
}


def get_backend_space_expansion(frontend_key: str) -> str:
    result = EXPANSION_FRONTEND_TO_BACKEND_MAPPING_SPACE.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid expansion key: {frontend_key}")
    return result


# SpaceFields
SPACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "Space_ID": "id",
    "Space_State": "state",
    "Creation_Time": "created_at",
    "End_Time": "ended_at",
    "Host_User_IDs": "host_ids",
    "Language": "lang",
    "Is_Ticketed": "is_ticketed",
    "Invited_User_IDs": "invited_user_ids",
    "Participant_Count": "participant_count",
    "Subscriber_Count": "subscriber_count",
    "Scheduled_Start_Time": "scheduled_start",
    "Speaker_User_IDs": "speaker_ids",
    "Start_Time": "started_at",
    "Space_Title": "title",
    "Topic_IDs": "topic_ids",
    "Last_Updated_Time": "updated_at",
}


def get_backend_space_field(frontend_key: str) -> str:
    result = SPACE_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid space field key: {frontend_key}")
    return result


# -------------- List Expansions -----------------

# ListExpansions
LIST_EXPANSION_FRONTEND_TO_BACKEND_MAPPING = {"List_Owner_ID": "owner_id"}


def get_backend_list_expansion(frontend_key: str) -> str:
    result = LIST_EXPANSION_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid list expansion key: {frontend_key}")
    return result


LIST_FIELDS_FRONTEND_TO_BACKEND_MAPPING = {
    "List_ID": "id",
    "List_Name": "name",
    "Creation_Date": "created_at",
    "Description": "description",
    "Follower_Count": "follower_count",
    "Member_Count": "member_count",
    "Is_Private": "private",
    "Owner_ID": "owner_id",
}


def get_backend_list_field(frontend_key: str) -> str:
    result = LIST_FIELDS_FRONTEND_TO_BACKEND_MAPPING.get(frontend_key)
    if result is None:
        raise KeyError(f"Invalid list field key: {frontend_key}")
    return result
