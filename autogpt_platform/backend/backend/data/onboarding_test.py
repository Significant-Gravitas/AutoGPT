from backend.data.onboarding import format_onboarding_for_extraction


def test_format_onboarding_for_extraction_basic():
    result = format_onboarding_for_extraction(
        user_name="John",
        user_role="Founder/CEO",
        pain_points=["Finding leads", "Email & outreach"],
    )
    assert "Q: What is your name?" in result
    assert "A: John" in result
    assert "Q: What best describes your role?" in result
    assert "A: Founder/CEO" in result
    assert "Q: What tasks are eating your time?" in result
    assert "Finding leads" in result
    assert "Email & outreach" in result


def test_format_onboarding_for_extraction_with_other():
    result = format_onboarding_for_extraction(
        user_name="Jane",
        user_role="Data Scientist",
        pain_points=["Research", "Building dashboards"],
    )
    assert "A: Jane" in result
    assert "A: Data Scientist" in result
    assert "Research, Building dashboards" in result
