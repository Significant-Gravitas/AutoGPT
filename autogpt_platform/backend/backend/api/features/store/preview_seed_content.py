"""The synthetic marketplace the preview seed publishes.

Every person here is made up: their names start with "Sample", their emails
are on the preview persona domain, and none of them has a password, so nobody
can sign in as them. Each listing is a small workflow built only from blocks
that compute without a model, credentials or the network, so anyone in a
preview can add one to their library and run it.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

EMAIL_DOMAIN = "previews.agpt.co"


class _Content(BaseModel):
    model_config = ConfigDict(frozen=True)


class SeedUser(_Content):
    key: str
    name: str
    username: str
    description: str
    links: tuple[str, ...] = ()
    featured: bool = False

    @property
    def email(self) -> str:
        return f"seed-{self.username}@{EMAIL_DOMAIN}"


class TemplateWorkflow(_Content):
    """One text input rendered through a Jinja template."""

    input_name: str
    input_title: str
    example: str
    template: str
    output_name: str
    long_input: bool = False


class CalculatorWorkflow(_Content):
    """One amount put through a single arithmetic step.

    The amount is entered as text: the number input holds whole numbers only,
    so it would drop the cents of a price before the calculator saw them.
    """

    input_name: str
    input_title: str
    example: str
    operation: Literal["Add", "Subtract", "Multiply", "Divide"]
    operand: float
    output_name: str


class ReplaceWorkflow(_Content):
    """A pasted text with one substring replaced throughout."""

    input_name: str
    input_title: str
    example: str
    old: str
    new: str
    output_name: str


class CountWorkflow(_Content):
    """A pasted list, one item per line, counted. Blank lines don't count."""

    input_name: str
    input_title: str
    example: str
    output_name: str


Workflow = TemplateWorkflow | CalculatorWorkflow | ReplaceWorkflow | CountWorkflow


class SeedListing(_Content):
    slug: str
    creator: str
    name: str
    sub_heading: str
    description: str
    categories: tuple[str, ...]
    workflow: Workflow
    # Star ratings, each from a different synthetic reviewer.
    reviews: tuple[int, ...] = ()
    featured: bool = False
    instructions: str | None = None
    recommended_schedule_cron: str | None = None


class SeedSkill(_Content):
    creator: str
    slug: str
    categories: tuple[str, ...]
    skill_md: str


CREATORS: tuple[SeedUser, ...] = (
    SeedUser(
        key="growth",
        name="Sample Growth Studio",
        username="sample-growth-studio",
        description=(
            "Synthetic creator for preview environments. Small marketing and "
            "sales workflows for teams that launch every week."
        ),
        links=("https://example.com/sample-growth-studio",),
        featured=True,
    ),
    SeedUser(
        key="ledger",
        name="Sample Ledger Works",
        username="sample-ledger-works",
        description=(
            "Synthetic creator for preview environments. Finance and operations "
            "helpers that do the arithmetic and the paperwork."
        ),
        links=("https://example.com/sample-ledger-works",),
    ),
    SeedUser(
        key="helpdesk",
        name="Sample Helpdesk Co",
        username="sample-helpdesk-co",
        description=(
            "Synthetic creator for preview environments. Support workflows for "
            "replying faster and keeping tickets tidy."
        ),
    ),
    SeedUser(
        key="research",
        name="Sample Research Desk",
        username="sample-research-desk",
        description=(
            "Synthetic creator for preview environments. Research and writing "
            "workflows that turn notes into something readable."
        ),
        links=("https://example.com/sample-research-desk",),
        featured=True,
    ),
    SeedUser(
        key="devtools",
        name="Sample Dev Workshop",
        username="sample-dev-workshop",
        description=(
            "Synthetic creator for preview environments. Developer workflows for "
            "commits, release notes and docs."
        ),
    ),
)

REVIEWERS: tuple[SeedUser, ...] = tuple(
    SeedUser(
        key=f"reviewer-{n}",
        name=f"Sample Reviewer {label}",
        username=f"sample-reviewer-{n}",
        description="Synthetic marketplace user for preview environments.",
    )
    for n, label in ((1, "One"), (2, "Two"), (3, "Three"))
)

# Approves the submissions, as a platform admin would.
ADMIN = SeedUser(
    key="admin",
    name="Sample Marketplace Admin",
    username="sample-marketplace-admin",
    description="Synthetic marketplace admin for preview environments.",
)

REVIEW_COMMENTS: dict[int, str] = {
    5: "Worked first time and saves me a few minutes every day.",
    4: "Does what it says. I tweaked the template a little.",
    3: "A useful starting point, but I edit the output each time.",
    2: "Too basic for my team's process.",
    1: "Not what I expected from the description.",
}

LISTINGS: tuple[SeedListing, ...] = (
    # --- Sample Growth Studio: marketing and sales -------------------------
    SeedListing(
        slug="social-media-post-planner",
        creator="growth",
        name="Social Media Post Planner",
        sub_heading="Turn one announcement into a week of social media posts",
        description=(
            "Paste a product announcement and get a five-day plan of social "
            "media posts, one per channel, ready to schedule."
        ),
        categories=("marketing", "content"),
        workflow=TemplateWorkflow(
            input_name="announcement",
            input_title="Announcement",
            example="Our new reporting dashboard is live for every workspace.",
            template=(
                "Monday (LinkedIn): {{ announcement }}\n"
                "Tuesday (X): Big news: {{ announcement }}\n"
                "Wednesday (Instagram): Behind the scenes of this week's "
                "launch: {{ announcement }}\n"
                "Thursday (Newsletter): In case you missed it: {{ announcement }}\n"
                "Friday (LinkedIn): Thanks for a great launch week. "
                "{{ announcement }}"
            ),
            output_name="post_plan",
            long_input=True,
        ),
        reviews=(5, 5, 4, 4),
        featured=True,
        instructions="Paste the announcement as you would publish it.",
        recommended_schedule_cron="0 9 * * 1",
    ),
    SeedListing(
        slug="campaign-brief-outline",
        creator="growth",
        name="Campaign Brief Outline",
        sub_heading="Start every campaign from the same one-page brief",
        description=(
            "Name the goal of a campaign and get a one-page brief with the "
            "audience, message, channels and success measure to fill in."
        ),
        categories=("marketing",),
        workflow=TemplateWorkflow(
            input_name="campaign_goal",
            input_title="Campaign goal",
            example="Grow trial sign-ups from small agencies",
            template=(
                "Campaign brief: {{ campaign_goal }}\n\n"
                "Audience: who needs to hear this?\n"
                "Message: the one sentence they should remember.\n"
                "Channels: where they already spend time.\n"
                "Success measure: the number that proves it worked.\n"
                "Deadline: when the campaign ends."
            ),
            output_name="brief",
        ),
        reviews=(4, 3),
    ),
    SeedListing(
        slug="utm-link-builder",
        creator="growth",
        name="UTM Link Builder",
        sub_heading="Tag a landing page for every channel in one go",
        description=(
            "Give it a landing page URL and get the same link tagged with UTM "
            "parameters for LinkedIn, the newsletter and paid search."
        ),
        categories=("marketing",),
        workflow=TemplateWorkflow(
            input_name="landing_page_url",
            input_title="Landing page URL",
            example="https://example.com/launch",
            # Parameters go before any #fragment, after a & when the URL
            # already has a query string.
            template=(
                "{% set url, hash, fragment = landing_page_url.partition('#') %}"
                "{% set sep = '&' if '?' in url else '?' %}"
                "LinkedIn: {{ url }}{{ sep }}utm_source=linkedin"
                "&utm_medium=social&utm_campaign=launch{{ hash }}{{ fragment }}\n"
                "Newsletter: {{ url }}{{ sep }}utm_source=newsletter"
                "&utm_medium=email&utm_campaign=launch{{ hash }}{{ fragment }}\n"
                "Paid search: {{ url }}{{ sep }}utm_source=google"
                "&utm_medium=cpc&utm_campaign=launch{{ hash }}{{ fragment }}"
            ),
            output_name="tagged_links",
        ),
        reviews=(5, 4, 5),
    ),
    SeedListing(
        slug="follow-up-email-drafter",
        creator="growth",
        name="Follow-up Email Drafter",
        sub_heading="Draft a polite follow-up email after a sales call",
        description=(
            "Paste your notes from a sales call and get a follow-up email that "
            "recaps the conversation and proposes the next step."
        ),
        categories=("sales",),
        workflow=TemplateWorkflow(
            input_name="call_notes",
            input_title="Call notes",
            example="Discussed the annual plan. They want a security review first.",
            template=(
                "Subject: Following up on our call\n\n"
                "Hi there,\n\n"
                "Thanks for your time today. Here is what we covered:\n"
                "{{ call_notes }}\n\n"
                "Shall we book thirty minutes next week for the next step?\n\n"
                "Best regards"
            ),
            output_name="email",
            long_input=True,
        ),
        reviews=(5, 4, 4),
        featured=True,
    ),
    SeedListing(
        slug="lead-generation-scorecard",
        creator="growth",
        name="Lead Generation Scorecard",
        sub_heading="Check a new lead against a simple fit checklist",
        description=(
            "Paste what you know about a new lead and get a lead generation "
            "scorecard to rate their fit before anyone books a demo."
        ),
        categories=("sales", "marketing"),
        workflow=TemplateWorkflow(
            input_name="lead_details",
            input_title="Lead details",
            example="Agency, 40 people, uses spreadsheets for reporting.",
            template=(
                "Lead generation scorecard\n\n"
                "Lead: {{ lead_details }}\n\n"
                "[ ] Matches our ideal company size\n"
                "[ ] Has the problem we solve today\n"
                "[ ] Has budget this quarter\n"
                "[ ] We know who decides"
            ),
            output_name="scorecard",
            long_input=True,
        ),
        reviews=(4, 4, 3, 5),
    ),
    SeedListing(
        slug="discount-price-calculator",
        creator="growth",
        name="Discount Price Calculator",
        sub_heading="Apply the standard 15% partner discount to a list price",
        description=(
            "Enter a list price and get the price after the standard 15% "
            "partner discount, so quotes stay consistent."
        ),
        categories=("sales", "finance"),
        workflow=CalculatorWorkflow(
            input_name="list_price",
            input_title="List price",
            example="1199.99",
            operation="Multiply",
            operand=0.85,
            output_name="discounted_price",
        ),
        reviews=(3, 4),
    ),
    # --- Sample Ledger Works: finance and operations -----------------------
    SeedListing(
        slug="invoice-total-with-tax",
        creator="ledger",
        name="Invoice Total with Tax",
        sub_heading="Add 8% sales tax to an invoice subtotal",
        description=(
            "Enter an invoice subtotal and get the total including 8% sales "
            "tax, rounded to the cent by your invoicing tool."
        ),
        categories=("finance",),
        workflow=CalculatorWorkflow(
            input_name="subtotal",
            input_title="Subtotal",
            example="2499.50",
            operation="Multiply",
            operand=1.08,
            output_name="total",
        ),
        reviews=(5, 5, 5),
        featured=True,
    ),
    SeedListing(
        slug="monthly-budget-splitter",
        creator="ledger",
        name="Monthly Budget Splitter",
        sub_heading="Split an annual budget into twelve equal months",
        description=(
            "Enter an annual budget and get the amount each month can spend if "
            "it is spread evenly across the year."
        ),
        categories=("finance", "operations"),
        workflow=CalculatorWorkflow(
            input_name="annual_budget",
            input_title="Annual budget",
            example="120000",
            operation="Divide",
            operand=12,
            output_name="monthly_budget",
        ),
        reviews=(4, 4),
    ),
    SeedListing(
        slug="expense-memo-formatter",
        creator="ledger",
        name="Expense Memo Formatter",
        sub_heading="Write the same memo line for every expense claim",
        description=(
            "Describe an expense and get a memo line in the format finance "
            "asks for, so claims stop bouncing back."
        ),
        categories=("finance",),
        workflow=TemplateWorkflow(
            input_name="expense",
            input_title="Expense",
            example="Team lunch with the Acme onboarding group",
            template="EXPENSE | {{ expense }} | business purpose: client work",
            output_name="memo",
        ),
        reviews=(3,),
    ),
    SeedListing(
        slug="calendar-invite-composer",
        creator="ledger",
        name="Calendar Invite Composer",
        sub_heading="Write a clear calendar invite with an agenda",
        description=(
            "Give it a meeting topic and get calendar invite text with an "
            "agenda, an owner and the decision the meeting should reach."
        ),
        categories=("operations",),
        workflow=TemplateWorkflow(
            input_name="meeting_topic",
            input_title="Meeting topic",
            example="Quarterly planning",
            template=(
                "Title: {{ meeting_topic }}\n\n"
                "Agenda:\n"
                "1. Where we are\n"
                "2. Options\n"
                "3. Decision and owners\n\n"
                "Decision needed: agree the plan for {{ meeting_topic }}."
            ),
            output_name="invite",
        ),
        reviews=(5, 4),
    ),
    SeedListing(
        slug="meeting-notes-cleanup",
        creator="ledger",
        name="Meeting Notes Cleanup",
        sub_heading="Normalise the bullet points in raw meeting notes",
        description=(
            "Paste notes with asterisk bullets and get them back with dashes, "
            "ready to paste into your team wiki."
        ),
        categories=("operations",),
        workflow=ReplaceWorkflow(
            input_name="notes",
            input_title="Notes",
            example="* Agreed the launch date\n* Sam owns the checklist",
            old="* ",
            new="- ",
            output_name="clean_notes",
        ),
        reviews=(4, 3, 4),
    ),
    # --- Sample Helpdesk Co: support ---------------------------------------
    SeedListing(
        slug="checklist-item-counter",
        creator="helpdesk",
        name="Checklist Item Counter",
        sub_heading="Count the items in a pasted checklist",
        description=(
            "Paste a checklist with one item per line and get the number of "
            "items, to size a handover before you start it."
        ),
        categories=("operations",),
        workflow=CountWorkflow(
            input_name="checklist",
            input_title="Checklist",
            example="Reset password\nUpdate billing contact\nClose ticket",
            output_name="item_count",
        ),
    ),
    SeedListing(
        slug="support-reply-drafter",
        creator="helpdesk",
        name="Support Reply Drafter",
        sub_heading="Answer a customer email with your standard reply",
        description=(
            "Paste the customer's question and get a friendly first reply to "
            "their email that sets expectations for a full answer."
        ),
        categories=("support",),
        workflow=TemplateWorkflow(
            input_name="customer_question",
            input_title="Customer question",
            example="How do I export my data to a spreadsheet?",
            template=(
                "Hi there,\n\n"
                "Thanks for reaching out about: {{ customer_question }}\n\n"
                "We are on it and will reply with a full answer within one "
                "business day.\n\n"
                "Best,\nThe support team"
            ),
            output_name="reply",
            long_input=True,
        ),
        reviews=(5, 5, 4, 5),
        featured=True,
    ),
    SeedListing(
        slug="ticket-summary-card",
        creator="helpdesk",
        name="Ticket Summary Card",
        sub_heading="Turn a support ticket into a one-line summary card",
        description=(
            "Paste a ticket's subject and get a summary card with status, "
            "owner and next step for the shift handover."
        ),
        categories=("support",),
        workflow=TemplateWorkflow(
            input_name="ticket_subject",
            input_title="Ticket subject",
            example="Customer cannot reset their password",
            template=(
                "TICKET: {{ ticket_subject }}\n"
                "Status: open | Owner: unassigned | Next step: reproduce"
            ),
            output_name="summary_card",
        ),
        reviews=(4,),
    ),
    SeedListing(
        slug="refund-after-fee-calculator",
        creator="helpdesk",
        name="Refund After Fee Calculator",
        sub_heading="Work out a refund after the 10% restocking fee",
        description=(
            "Enter the amount a customer paid and get the refund after the "
            "standard 10% restocking fee."
        ),
        categories=("support", "finance"),
        workflow=CalculatorWorkflow(
            input_name="amount_paid",
            input_title="Amount paid",
            example="79.90",
            operation="Multiply",
            operand=0.9,
            output_name="refund",
        ),
        reviews=(3, 4, 2),
    ),
    # --- Sample Research Desk: research and content ------------------------
    SeedListing(
        slug="research-brief-outline",
        creator="research",
        name="Research Brief Outline",
        sub_heading="Frame a research question before you start digging",
        description=(
            "Write down a research question and get a brief that pins down "
            "scope, sources and what a good answer looks like."
        ),
        categories=("research",),
        workflow=TemplateWorkflow(
            input_name="research_question",
            input_title="Research question",
            example="Which onboarding emails do new users actually open?",
            template=(
                "Research brief\n\n"
                "Question: {{ research_question }}\n"
                "Scope: what is in and out of bounds.\n"
                "Sources: where the evidence lives.\n"
                "Done when: what a good answer looks like."
            ),
            output_name="brief",
        ),
        reviews=(5, 4, 5),
        featured=True,
    ),
    SeedListing(
        slug="news-digest-formatter",
        creator="research",
        name="News Digest Formatter",
        sub_heading="Number the news you summarize into one daily digest",
        description=(
            "Paste the news items you summarize each morning, one per line, and "
            "get them back as a numbered digest ready to send to the team."
        ),
        categories=("research", "content"),
        workflow=TemplateWorkflow(
            input_name="headlines",
            input_title="News items",
            example="Rates held steady\nNew privacy rules proposed",
            template=(
                "Daily digest\n"
                "{% for item in headlines.splitlines() if item.strip() %}\n"
                "{{ loop.index }}. {{ item.strip() }}{% endfor %}\n\n"
                "Compiled for this morning's read."
            ),
            output_name="digest",
            long_input=True,
        ),
        reviews=(4, 4, 5),
        recommended_schedule_cron="0 8 * * 1-5",
    ),
    SeedListing(
        slug="survey-response-counter",
        creator="research",
        name="Survey Response Counter",
        sub_heading="Count the responses in a pasted survey export",
        description=(
            "Paste a survey export with one response per line and get the "
            "number of responses, before you start reading them."
        ),
        categories=("research",),
        workflow=CountWorkflow(
            input_name="responses",
            input_title="Responses",
            example="Loved it\nToo slow\nWould recommend",
            output_name="response_count",
        ),
        reviews=(3, 4),
    ),
    SeedListing(
        slug="blog-post-outline",
        creator="research",
        name="Blog Post Outline",
        sub_heading="Lay out a blog post from a working title",
        description=(
            "Give it a working title and get a blog post outline with a hook, "
            "three sections and a call to action."
        ),
        categories=("content", "marketing"),
        workflow=TemplateWorkflow(
            input_name="working_title",
            input_title="Working title",
            example="Five ways to cut your reporting time",
            template=(
                "# {{ working_title }}\n\n"
                "Hook: why this matters now.\n"
                "## Section one\n## Section two\n## Section three\n"
                "Call to action: what the reader should do next."
            ),
            output_name="outline",
        ),
        reviews=(4, 5, 4),
    ),
    SeedListing(
        slug="headline-title-caser",
        creator="research",
        name="Headline Title Caser",
        sub_heading="Put every headline into consistent title case",
        description=(
            "Paste a headline and get it back in title case, so a page of "
            "headlines reads consistently."
        ),
        categories=("content",),
        workflow=TemplateWorkflow(
            input_name="headline",
            input_title="Headline",
            example="five ways to cut your reporting time",
            template="{{ headline | title }}",
            output_name="title_cased",
        ),
        reviews=(4, 3),
    ),
    # --- Sample Dev Workshop: development ----------------------------------
    SeedListing(
        slug="release-notes-formatter",
        creator="devtools",
        name="Release Notes Formatter",
        sub_heading="Turn a list of merged changes into release notes",
        description=(
            "Paste the changes merged since the last release and get release "
            "notes with a heading and an upgrade reminder."
        ),
        categories=("content", "development"),
        workflow=TemplateWorkflow(
            input_name="changes",
            input_title="Changes",
            example="Faster exports\nFixed the login redirect",
            template=(
                "## What's new\n\n{{ changes }}\n\n"
                "Upgrade at your convenience; nothing needs migrating."
            ),
            output_name="release_notes",
            long_input=True,
        ),
        reviews=(5, 4),
    ),
    SeedListing(
        slug="commit-message-template",
        creator="devtools",
        name="Commit Message Template",
        sub_heading="Write conventional commit messages from a summary",
        description=(
            "Describe a change in a few words and get a conventional commit "
            "message with a body for the details."
        ),
        categories=("development",),
        workflow=TemplateWorkflow(
            input_name="change_summary",
            input_title="Change summary",
            example="add CSV export to the reports page",
            template=(
                "feat: {{ change_summary }}\n\n"
                "Why: the problem this solves.\n"
                "How: the approach, in a sentence or two."
            ),
            output_name="commit_message",
        ),
        reviews=(5, 4, 5),
        featured=True,
    ),
    SeedListing(
        slug="changed-file-counter",
        creator="devtools",
        name="Changed File Counter",
        sub_heading="Count the files in a pasted diff summary",
        description=(
            "Paste a diff summary with one file per line and get the number of "
            "files changed, to size a code review."
        ),
        categories=("development",),
        workflow=CountWorkflow(
            input_name="changed_files",
            input_title="Changed files",
            example="src/app.py\nsrc/export.py\ntests/test_export.py",
            output_name="file_count",
        ),
    ),
    SeedListing(
        slug="api-endpoint-doc-stub",
        creator="devtools",
        name="API Endpoint Doc Stub",
        sub_heading="Start an endpoint's documentation from its route",
        description=(
            "Give it a route such as GET /reports and get a documentation stub "
            "with auth, parameters and an example response to fill in."
        ),
        categories=("development",),
        workflow=TemplateWorkflow(
            input_name="route",
            input_title="Route",
            example="GET /reports",
            template=(
                "## {{ route }}\n\n"
                "Auth: required.\n"
                "Parameters: none yet.\n"
                "Example response: `{}`"
            ),
            output_name="doc_stub",
        ),
        reviews=(4, 5, 4),
    ),
)

SKILLS: tuple[SeedSkill, ...] = (
    SeedSkill(
        creator="growth",
        slug="sample-email-subject-lines",
        categories=("sales", "marketing"),
        skill_md="""---
name: sample-email-subject-lines
description: Write five subject lines for a sales email, from the safest to the boldest.
triggers: ["subject line", "email subject"]
---

# Email subject lines

When asked for subject lines for an email:

1. Read the email body first and find its single most useful promise.
2. Write five subject lines under 50 characters, from the safest to the boldest.
3. Never use all caps or more than one exclamation mark.
""",
    ),
    SeedSkill(
        creator="ledger",
        slug="sample-month-end-checklist",
        categories=("finance",),
        skill_md="""---
name: sample-month-end-checklist
description: Walk through a short month-end close checklist and flag what is missing.
triggers: ["month end", "close the books"]
---

# Month-end checklist

Go through these in order and say which are done, which are missing and who owns each:

1. Bank accounts reconciled.
2. Open invoices reviewed.
3. Accruals booked.
4. Summary sent to the finance lead.
""",
    ),
    SeedSkill(
        creator="helpdesk",
        slug="sample-support-tone-guide",
        categories=("support",),
        skill_md="""---
name: sample-support-tone-guide
description: Rewrite a support reply so it is warm, short and says what happens next.
triggers: ["support reply", "customer reply"]
---

# Support tone guide

When rewriting a reply to a customer:

- Open by thanking them for the detail they gave.
- Keep it under 120 words.
- End with exactly what happens next and when.
""",
    ),
    SeedSkill(
        creator="research",
        slug="sample-source-quality-check",
        categories=("research",),
        skill_md="""---
name: sample-source-quality-check
description: Rate each source in a list for recency, authority and relevance before citing it.
triggers: ["check sources", "source quality"]
---

# Source quality check

For each source, give three short ratings and one sentence of reasoning:

- Recency: is it recent enough for the claim?
- Authority: who published it, and why should we trust them?
- Relevance: does it support the exact claim, or only a nearby one?
""",
    ),
    SeedSkill(
        creator="devtools",
        slug="sample-code-review-checklist",
        categories=("development",),
        skill_md="""---
name: sample-code-review-checklist
description: Review a diff against a short checklist and list concrete findings.
triggers: ["review this diff", "code review"]
---

# Code review checklist

Check the diff for each of these and list findings with file and line:

1. Behaviour changes without a test.
2. Error paths that swallow failures.
3. Names that no longer match what the code does.
""",
    ),
)
