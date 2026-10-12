"""Edition 1 design contracts across the complete transactional inventory."""

import re
from html.parser import HTMLParser
from typing import get_args

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import (
    DeliveryStream,
    TrialUpdateData,
    VerdictData,
    get_delivery_stream,
)
from backend.notifications.design_system_fixtures import URLS, EmailScenario, scenarios
from backend.notifications.renderer import RenderedEmail, render

SCENARIOS = scenarios()
OTTO_PATH = "/autogpt-characters/v1.1/otto/neutral/256.png"
ADDRESS = "3rd Floor, 1 Ashley Road, Altrincham, WA14 2DT, UK"


@pytest.fixture(params=SCENARIOS, ids=lambda scenario: scenario.name)
def rendered(request) -> tuple[EmailScenario, RenderedEmail, "EmailDocument"]:
    scenario = request.param
    email = render(scenario.notification_type, scenario.data, "sam@example.com", URLS)
    document = EmailDocument()
    document.feed(email.html)
    return scenario, email, document


def test_every_transactional_variant_uses_the_shared_frame(rendered):
    _, email, document = rendered
    assert email.html.startswith("<!doctype html>")
    assert len(email.html.encode("utf-8")) < 60 * 1024
    assert email.preheader and email.text
    assert "Geist," in document.elements("body")[0]["style"]
    assert "Geist+Mono" in email.html
    assert any(table.get("width") == "560" for table in document.elements("table"))
    for table in document.elements("table"):
        assert table.get("role") == "presentation"
        assert table.get("bgcolor"), table
    assert ADDRESS in document.text
    assert "<svg" not in email.html.lower()
    assert "data:image" not in email.html.lower()
    assert "Poppins" not in email.html


def test_masthead_and_otto_survive_email_image_restrictions(rendered):
    scenario, _, document = rendered
    images = document.elements("img")
    assert len(images) == 2
    logo, otto = images
    assert logo["src"].endswith("/email/logo-light.png")
    assert (logo["width"], logo["height"], logo["alt"]) == ("100", "45", "AutoGPT")
    assert otto["src"].endswith(OTTO_PATH)
    size = "80" if scenario.notification_type == NotificationType.OPS else "160"
    assert (otto["width"], otto["height"]) == (size, size)
    assert otto["alt"] == "Otto, your personal Head of AI"
    for image in images:
        assert image["src"].startswith("https://")
        assert re.search(r"\.(png|jpe?g)$", image["src"], re.I)
    assert any(link == "https://agpt.co" for link in document.links)
    assert document.text.strip()


def test_authored_copy_uses_workflow_vocabulary_and_plain_punctuation(rendered):
    _, email, document = rendered
    for part in (email.subject, email.preheader, document.text, email.text):
        assert "—" not in part
        assert "–" not in part
        assert not re.search(r"\bagents?\b", part, re.I)
    for banned in ("Summary", "Report", "Newsletter"):
        assert banned not in email.subject


def test_button_counts_follow_explicit_family_exceptions(rendered):
    scenario, _, document = rendered
    buttons = [
        element
        for element in document.elements("td")
        if element.get("bgcolor", "").upper() == "#6144DF"
    ]
    if scenario.notification_type == NotificationType.SUBSCRIPTION_CANCELLED:
        assert not buttons
        assert URLS.billing in document.links
    elif scenario.notification_type == NotificationType.BRIEFING:
        assert len(buttons) <= 1
        if scenario.name == "briefing-attention-overflow":
            assert len(buttons) == 1
    else:
        assert len(buttons) == 1


def test_footers_match_the_delivery_stream(rendered):
    scenario, email, document = rendered
    stream = get_delivery_stream(scenario.notification_type)
    if stream == DeliveryStream.PRODUCT:
        assert URLS.unsubscribe in document.links
        assert URLS.unsubscribe in email.text
        assert URLS.discord in document.links
        assert URLS.discord in email.text
    elif stream == DeliveryStream.BILLING:
        assert "service message" in document.text
        assert URLS.prefs in document.links
        assert URLS.prefs in email.text
        assert URLS.unsubscribe not in document.links
        assert URLS.unsubscribe not in email.text
    else:
        assert "Not customer-facing" in document.text
        assert URLS.unsubscribe not in document.links
        assert URLS.discord not in document.links


@pytest.mark.parametrize(
    "scenario",
    [s for s in SCENARIOS if s.notification_type == NotificationType.BRIEFING],
    ids=lambda scenario: scenario.name,
)
def test_briefing_preserves_signed_cadence_links(scenario):
    email = render(scenario.notification_type, scenario.data, "sam@example.com", URLS)
    current = scenario.data.model_dump()["period"]["frequency"]
    for choice, url in URLS.volume.items():
        if choice != current:
            assert url in email.html
        assert url in email.text
    assert re.search(
        r"<(?:strong|span)[^>]*font-weight:700[^>]*>"
        + current.capitalize()
        + r"</(?:strong|span)>",
        email.html,
    )
    assert "?f=" not in email.html
    for label in ("Daily", "Weekly", "Monthly", "Alerts only", "Pause"):
        assert label in email.html


def test_reviewer_feedback_is_complete_and_inert():
    feedback = '<script>alert("example")</script> Keep the user’s dash — unchanged.'
    data = VerdictData(
        outcome="changes",
        agent_name="Example <workflow>",
        version=1,
        reviewer_name="Alex",
        reviewed_at_label="3 Oct 2026",
        comments=feedback,
        resubmit_url="https://platform.example/edit",
    )
    email = render(NotificationType.VERDICT, data, "sam@example.com", URLS)
    document = EmailDocument()
    document.feed(email.html)
    assert not document.elements("script")
    assert feedback in document.text
    assert feedback in email.text
    assert "Example <workflow>" in document.text


def test_ops_keeps_customer_email_distinct_from_internal_recipient():
    scenario = next(s for s in SCENARIOS if s.name == "ops-request")
    email = render(
        scenario.notification_type, scenario.data, "refunds@example.com", URLS
    )
    assert "sam@example.com" in email.html
    assert "sam@example.com" in email.text


def test_scenarios_cover_every_notification_type_and_trial_state():
    assert {s.notification_type for s in SCENARIOS} == set(NotificationType)
    assert {
        s.name.removeprefix("trial-")
        for s in SCENARIOS
        if s.notification_type == NotificationType.TRIAL_UPDATE
    } == set(get_args(TrialUpdateData.model_fields["kind"].annotation))


class EmailDocument(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str]]] = []
        self._text: list[str] = []
        self._ignored: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.tags.append((tag, {key: value or "" for key, value in attrs}))
        if tag in ("head", "style", "script"):
            self._ignored.append(tag)

    def handle_endtag(self, tag: str):
        if self._ignored and self._ignored[-1] == tag:
            self._ignored.pop()

    def handle_data(self, data: str):
        if not self._ignored:
            self._text.append(data)

    def elements(self, tag: str) -> list[dict[str, str]]:
        return [attributes for name, attributes in self.tags if name == tag]

    @property
    def text(self) -> str:
        return " ".join(" ".join(self._text).split())

    @property
    def links(self) -> list[str]:
        return [attributes.get("href", "") for attributes in self.elements("a")]
