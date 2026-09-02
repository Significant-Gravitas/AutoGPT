"""The recruiter: spot a run of Autopilot-self-handled tasks that a hired
expert template already covers, and recommend the hire.

Pure keyword matching over template roles/skills — deterministic and
cheap enough for briefing-compose time. A category here is simply the
template whose vocabulary the task titles/specs keep hitting.
"""

from datetime import timedelta

from backend.api.features.experts.models import Expert
from backend.api.features.tasks.models import DelegatedTask
from backend.copilot.briefing.models import BriefingHireItem

RECRUITER_WINDOW = timedelta(days=14)

# How many same-category self-handled tasks it takes before Autopilot
# doing the work itself starts to look like a missing hire.
RECRUITER_TASK_THRESHOLD = 3

_MIN_TOKEN_LENGTH = 4


def compose_hire_items(
    autopilot_tasks: list[DelegatedTask],
    templates: list[Expert],
    hired: list[Expert],
) -> list[BriefingHireItem]:
    """At most one recommendation: the not-yet-hired template whose
    vocabulary matched the most recent self-handled tasks, once it clears
    the threshold."""
    hired_template_ids = {
        expert.source_template_id for expert in hired if expert.source_template_id
    }
    best: BriefingHireItem | None = None
    for template in templates:
        if template.id in hired_template_ids:
            continue
        vocabulary = _template_vocabulary(template)
        if not vocabulary:
            continue
        matched = [task for task in autopilot_tasks if _matches(task, vocabulary)]
        if len(matched) < RECRUITER_TASK_THRESHOLD:
            continue
        if best is None or len(matched) > best.task_count:
            best = BriefingHireItem(
                template_id=template.id,
                name=template.name,
                role=template.role,
                task_count=len(matched),
                example_titles=[task.title for task in matched[:3]],
            )
    return [best] if best else []


def _template_vocabulary(template: Expert) -> set[str]:
    tokens: set[str] = set()
    for phrase in (template.role, *template.skills):
        tokens.update(
            token
            for token in phrase.lower().replace("-", " ").split()
            if len(token) >= _MIN_TOKEN_LENGTH
        )
    return tokens


def _matches(task: DelegatedTask, vocabulary: set[str]) -> bool:
    text = f"{task.title} {task.spec}".lower()
    words = set(text.replace("-", " ").split())
    return bool(words & vocabulary)
