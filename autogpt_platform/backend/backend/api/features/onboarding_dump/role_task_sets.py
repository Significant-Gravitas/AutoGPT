"""The role starter sets: five tasks per wizard role, first one first.

Each names the template role whose expert takes it (None for Otto) and, for
the first, a ``why``. Copy, not logic: ``role_tasks`` builds the plan.
"""

from pydantic import BaseModel

OTHER_ROLE = "Other"


class RoleTask(BaseModel):
    title: str
    prompt: str
    icon: str
    owner_role: str | None = None
    why: str | None = None


ROLE_FALLBACK_TASKS: dict[str, list[RoleTask]] = {
    "Founder/CEO": [
        RoleTask(
            title="Build a list of 25 companies that look like your best customers",
            owner_role="sales",
            why="Most founders start here.",
            prompt="Build me a list of 25 companies that look like my best customers. Ask me who my best customers are today, then give me the list with a contact and one reason each.",
            icon="target",
        ),
        RoleTask(
            title="Draft next week's posts and a newsletter in your voice",
            owner_role="marketing",
            why="So something goes out while you build.",
            prompt="Draft next week's social posts and one newsletter in my voice. Ask me what I publish and where, then show me the first drafts.",
            icon="megaphone",
        ),
        RoleTask(
            title="Set up a Monday summary of your numbers, pulled from your tools",
            why="Until you hire someone for it.",
            prompt="Set up a Monday summary of the numbers I care about. Ask me which tools they live in and build the first one so I can see the format.",
            icon="chart-bar",
        ),
        RoleTask(
            title="Write a one-page pitch for your next investor or partner call",
            prompt="Write a one-page pitch for my next investor or partner conversation. Ask me three questions about the business first.",
            icon="file-text",
        ),
        RoleTask(
            title="Research three competitors and what they charge",
            owner_role="marketing",
            prompt="Research my three closest competitors: what they sell, what they charge and how they talk about it. Ask me who they are if you need to.",
            icon="magnifying-glass",
        ),
    ],
    "Operations": [
        RoleTask(
            title="Turn your weekly checklist into a tracker that chases itself",
            owner_role="ops",
            why="The thing that comes back every week.",
            prompt="Turn my weekly operations checklist into a tracker that reminds the right people. Ask me what is on it and who owns what.",
            icon="calendar-check",
        ),
        RoleTask(
            title="Draft the vendor follow-ups you send every week",
            owner_role="ops",
            prompt="Draft the follow-up messages I send to vendors every week. Ask me who they are and what I usually chase.",
            icon="envelope",
        ),
        RoleTask(
            title="Build the report you assemble by hand so it builds itself",
            owner_role="ops",
            prompt="Build the weekly report I put together by hand. Ask me what goes in it and where the numbers come from.",
            icon="chart-bar",
        ),
        RoleTask(
            title="Write the three procedures your team keeps asking you about",
            prompt="Write up the three procedures my team keeps asking me about. Ask me what they are and I will talk you through each.",
            icon="file-text",
        ),
        RoleTask(
            title="Clean up the spreadsheet you live in",
            owner_role="ops",
            prompt="Help me clean up the spreadsheet I live in. Ask me what it tracks and what is wrong with it.",
            icon="sparkle",
        ),
    ],
    "Sales/BD": [
        RoleTask(
            title="Build a list of 25 prospects that match your best customers",
            owner_role="sales",
            why="A clean list is where sales weeks start.",
            prompt="Build me a list of 25 prospects that look like my best customers, with a contact and one reason each. Ask me about my best customers first.",
            icon="target",
        ),
        RoleTask(
            title="Write the follow-up sequence you never get to",
            owner_role="sales",
            prompt="Write a three-step follow-up sequence for leads that went quiet, in my voice. Ask me what I sell and what usually stalls.",
            icon="envelope",
        ),
        RoleTask(
            title="Prep a one-page brief before each meeting on your calendar",
            owner_role="sales",
            prompt="Before each sales meeting, prepare a one-page brief on the company and the person. Ask me which calendar to watch.",
            icon="calendar-check",
        ),
        RoleTask(
            title="Keep your CRM clean without you touching it",
            owner_role="sales",
            prompt="Help me keep my CRM clean: stale deals, missing fields, duplicate contacts. Ask me which CRM I use.",
            icon="users",
        ),
        RoleTask(
            title="Research five accounts and what changed for them this month",
            prompt="Research five accounts I care about and tell me what changed for them this month. Ask me which ones.",
            icon="magnifying-glass",
        ),
    ],
    "Marketing": [
        RoleTask(
            title="Draft next week's posts in your voice, ready to edit",
            owner_role="marketing",
            why="The backlog most marketers carry.",
            prompt="Draft next week's posts in my voice. Ask me what I publish, where, and how often, then show me the drafts.",
            icon="megaphone",
        ),
        RoleTask(
            title="Write the newsletter that keeps slipping",
            owner_role="marketing",
            prompt="Write the next newsletter I keep putting off. Ask me who it goes to and what happened recently.",
            icon="envelope",
        ),
        RoleTask(
            title="Watch three competitors and send a Monday note on what they did",
            owner_role="marketing",
            prompt="Watch three competitors and send me a short Monday note on what they changed. Ask me who to track.",
            icon="bell",
        ),
        RoleTask(
            title="Pull your channel numbers into one weekly report",
            owner_role="marketing",
            prompt="Pull the numbers I report every week into one report. Ask me which channels and where the numbers live.",
            icon="chart-bar",
        ),
        RoleTask(
            title="Turn one long piece into five short ones",
            prompt="Take one long piece I already have and turn it into five short posts. Ask me for the piece.",
            icon="file-text",
        ),
    ],
    "Product/PM": [
        RoleTask(
            title="Turn this week's customer feedback into a ranked list",
            owner_role="project management",
            why="Feedback piles up faster than anyone reads it.",
            prompt="Turn the customer feedback I collected this week into a ranked list of themes. Ask me where the feedback lives.",
            icon="users",
        ),
        RoleTask(
            title="Draft the spec you have been putting off",
            prompt="Draft the spec I have been putting off. Ask me what the feature is and who it is for, then write the first version.",
            icon="file-text",
        ),
        RoleTask(
            title="Watch three competitors and tell you what they shipped",
            owner_role="marketing",
            prompt="Watch three competitors and tell me what they shipped each week. Ask me who to track.",
            icon="bell",
        ),
        RoleTask(
            title="Build your Monday metrics summary",
            owner_role="ops",
            prompt="Build a Monday summary of the product metrics I check. Ask me which ones and where they live.",
            icon="chart-bar",
        ),
        RoleTask(
            title="Write the release notes from this week's changes",
            prompt="Write release notes for what shipped this week. Ask me what changed or where to look.",
            icon="newspaper",
        ),
    ],
    "Engineering": [
        RoleTask(
            title="Write this week's status update from what shipped",
            why="The Friday hour most engineers lose.",
            prompt="Write my weekly status update from what shipped. Ask me what changed or where to look.",
            icon="newspaper",
        ),
        RoleTask(
            title="Draft the docs your team keeps asking for",
            prompt="Draft the documentation my team keeps asking for. Ask me what it covers and who reads it.",
            icon="file-text",
        ),
        RoleTask(
            title="Watch the tools and libraries you depend on for changes",
            prompt="Watch the tools and libraries I depend on and tell me when something important changes. Ask me which ones.",
            icon="bell",
        ),
        RoleTask(
            title="Answer the questions customers keep sending",
            prompt="Help me answer the questions customers keep sending. Ask me what they ask and where the answers live.",
            icon="chats",
        ),
        RoleTask(
            title="Build the report your manager asks for every week",
            owner_role="ops",
            prompt="Build the weekly report my manager asks for. Ask me what goes in it.",
            icon="chart-bar",
        ),
    ],
    "HR/People": [
        RoleTask(
            title="Screen this week's applicants against the role",
            owner_role="recruiting",
            why="The pile that grows every day.",
            prompt="Screen this week's applicants against the role. Ask me for the job description and where applications land.",
            icon="users",
        ),
        RoleTask(
            title="Draft the job post for the role you are hiring",
            owner_role="recruiting",
            prompt="Draft the job post for the role I am hiring. Ask me about the role first.",
            icon="file-text",
        ),
        RoleTask(
            title="Schedule interviews without the back and forth",
            owner_role="executive assistant",
            prompt="Take over scheduling interviews. Ask me which calendar and who is involved.",
            icon="calendar-check",
        ),
        RoleTask(
            title="Write the first-week onboarding plan for your next hire",
            prompt="Write a first-week onboarding plan for my next hire. Ask me about the role and the team.",
            icon="calendar-check",
        ),
        RoleTask(
            title="Pull the people numbers you report every month",
            owner_role="ops",
            prompt="Pull the people numbers I report every month into one summary. Ask me which ones.",
            icon="chart-bar",
        ),
    ],
    "Other": [
        RoleTask(
            title="Find the two tasks eating your week and hand them off",
            why="Start with what costs you the most time.",
            prompt="Help me find the two tasks eating my week and hand them off. Ask me what a normal week looks like.",
            icon="lightning",
        ),
        RoleTask(
            title="Get a sourced research brief on anything in minutes",
            prompt="I need research support. Let me give you a topic and send back a short brief with sources. Ask me what I am trying to learn about.",
            icon="magnifying-glass",
        ),
        RoleTask(
            title="Draft your next post, email or report in your voice",
            prompt="Draft the next thing I need to write in my voice. Ask me what it is and who it is for.",
            icon="megaphone",
        ),
        RoleTask(
            title="Set up a Monday summary of what matters to you",
            prompt="Set up a Monday summary of what matters to me. Ask me what to pull from.",
            icon="calendar-check",
        ),
        RoleTask(
            title="Watch something for you and tell you when it changes",
            prompt="Watch something for me and tell me when it changes. Ask me what.",
            icon="bell",
        ),
    ],
}
