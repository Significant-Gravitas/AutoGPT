from backend.util.media_generation_guidance import (
    IMAGE_GENERATION_LEADERBOARD_URL,
    IMAGE_GENERATION_MODEL_SELECTION_GUIDANCE,
    VIDEO_GENERATION_LEADERBOARD_URL,
    VIDEO_GENERATION_MODEL_SELECTION_GUIDANCE,
    image_generation_failure_message,
    response_detail,
    video_generation_failure_message,
)


def test_image_generation_guidance_uses_arena_leaderboard():
    assert (
        IMAGE_GENERATION_LEADERBOARD_URL == "https://arena.ai/leaderboard/text-to-image"
    )
    assert IMAGE_GENERATION_LEADERBOARD_URL in IMAGE_GENERATION_MODEL_SELECTION_GUIDANCE
    assert (
        "unless a specific model is requested"
        in IMAGE_GENERATION_MODEL_SELECTION_GUIDANCE
    )


def test_video_generation_guidance_uses_arena_leaderboard():
    assert (
        VIDEO_GENERATION_LEADERBOARD_URL == "https://arena.ai/leaderboard/text-to-video"
    )
    assert VIDEO_GENERATION_LEADERBOARD_URL in VIDEO_GENERATION_MODEL_SELECTION_GUIDANCE
    assert (
        "unless a specific model is requested"
        in VIDEO_GENERATION_MODEL_SELECTION_GUIDANCE
    )


def test_image_generation_failure_message_suggests_fallback_model():
    message = image_generation_failure_message("Provider unavailable")

    assert message.startswith("Provider unavailable.")
    assert "try another image generation model" in message
    assert "tell them the selected model appears to be down or unavailable" in message


def test_video_generation_failure_message_suggests_fallback_model():
    message = video_generation_failure_message("Provider unavailable")

    assert message.startswith("Provider unavailable.")
    assert "try another video generation model or block" in message
    assert "tell them the selected model appears to be down or unavailable" in message


def test_image_generation_failure_message_does_not_suggest_safety_bypass():
    message = image_generation_failure_message(
        "Content was flagged as sensitive by the model provider"
    )

    assert message == "Content was flagged as sensitive by the model provider."
    assert "try another image generation model" not in message
    assert "down or unavailable" not in message


def test_video_generation_failure_message_does_not_suggest_policy_bypass():
    message = video_generation_failure_message("Prompt violates provider safety policy")

    assert message == "Prompt violates provider safety policy."
    assert "try another video generation model or block" not in message
    assert "down or unavailable" not in message


def test_image_generation_failure_message_does_not_suggest_auth_fallback():
    message = image_generation_failure_message("Invalid API key")

    assert message == "Invalid API key."
    assert "try another image generation model" not in message
    assert "down or unavailable" not in message


def test_image_generation_failure_message_is_idempotent():
    once = image_generation_failure_message("Provider unavailable")
    twice = image_generation_failure_message(once)

    assert once == twice


def test_video_generation_failure_message_is_idempotent():
    once = video_generation_failure_message("Provider unavailable")
    twice = video_generation_failure_message(once)

    assert once == twice


def test_image_generation_failure_message_treats_status_503_as_fallback():
    message = image_generation_failure_message("503")

    assert "try another image generation model" in message


def test_image_generation_failure_message_does_not_treat_embedded_503_as_status():
    assert "try another image generation model" not in image_generation_failure_message(
        "1503"
    )
    assert "try another image generation model" not in image_generation_failure_message(
        "x503y"
    )


def test_image_generation_failure_message_matches_fallback_markers_case_insensitively():
    message = image_generation_failure_message("Service UNAVAILABLE")

    assert "try another image generation model" in message


def test_image_generation_failure_message_does_not_fallback_on_status_401():
    message = image_generation_failure_message("401")

    assert message == "401."
    assert "try another image generation model" not in message


def test_image_generation_failure_message_uses_default_for_blank_input():
    assert image_generation_failure_message("") == "Image generation failed."
    assert image_generation_failure_message("   ") == "Image generation failed."


def test_video_generation_failure_message_uses_default_for_blank_input():
    assert video_generation_failure_message("") == "Video generation failed."
    assert video_generation_failure_message("   ") == "Video generation failed."


def test_no_fallback_marker_wins_over_fallback_status():
    message = image_generation_failure_message(
        "503 Service Unavailable: content was flagged by moderation"
    )

    assert "try another image generation model" not in message
    assert "down or unavailable" not in message


def test_hard_no_fallback_marker_wins_over_fallback_marker():
    message = image_generation_failure_message(
        "Timed out waiting for moderation review"
    )

    assert "try another image generation model" not in message
    assert "down or unavailable" not in message


def test_fallback_marker_wins_over_soft_no_fallback_marker():
    message = image_generation_failure_message("Timed out due to invalid prompt")

    assert "try another image generation model" in message


def test_fallback_marker_wins_over_broad_capacity_message():
    message = image_generation_failure_message(
        "Insufficient capacity, please retry later (server error)"
    )

    assert "try another image generation model" in message


def test_insufficient_credits_still_blocks_fallback():
    message = image_generation_failure_message("Insufficient credits to run this model")

    assert "try another image generation model" not in message


def test_broad_validation_word_no_longer_blocks_fallback():
    message = image_generation_failure_message("Internal validation 503")

    assert "try another image generation model" in message


def test_image_generation_failure_message_suggests_fallback_for_unprocessable_output():
    message = image_generation_failure_message(
        "Unable to process result. Please contact us with the models and inputs used"
    )

    assert "try another image generation model" in message


def test_fallback_marker_wins_over_no_fallback_status():
    message = image_generation_failure_message("Model unavailable (404)")

    assert "try another image generation model" in message


def test_status_408_triggers_fallback():
    message = image_generation_failure_message("HTTP 408 Request Timeout")

    assert "try another image generation model" in message


def test_response_detail_picks_first_truthy_field():
    assert response_detail({"error": "boom"}) == "boom"
    assert response_detail({"message": "oops"}) == "oops"
    assert response_detail({"detail": "details"}) == "details"
    assert response_detail({"status": "FAILED"}) == "FAILED"


def test_response_detail_prefers_error_over_others():
    assert response_detail({"error": "primary", "message": "secondary"}) == "primary"


def test_response_detail_returns_none_for_empty_response():
    assert response_detail({}) is None
    assert response_detail({"error": ""}) is None
    assert response_detail({"unrelated": "value"}) is None


def test_response_detail_stringifies_non_string_values():
    assert response_detail({"status": 500}) == "500"


def test_response_detail_recurses_into_nested_dict():
    assert response_detail({"error": {"code": 500, "message": "boom"}}) == "boom"


def test_response_detail_skips_dict_with_no_useful_fields():
    assert (
        response_detail({"error": {"unrelated": "x"}, "message": "fallback"})
        == "fallback"
    )
