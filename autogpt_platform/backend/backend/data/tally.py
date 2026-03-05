"""Tally form integration: cache submissions, match by email, extract business understanding."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncOpenAI

from backend.data.redis_client import get_redis_async
from backend.data.understanding import (
    BusinessUnderstandingInput,
    get_business_understanding,
    upsert_business_understanding,
)
from backend.util.request import Requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

TALLY_API_BASE = "https://api.tally.so"
_settings = Settings()
TALLY_FORM_ID = _settings.secrets.tally_form_id

# Redis key templates
_EMAIL_INDEX_KEY = "tally:form:{form_id}:email_index"
_QUESTIONS_KEY = "tally:form:{form_id}:questions"
_LAST_FETCH_KEY = "tally:form:{form_id}:last_fetch"

# TTLs — keep aligned so last_fetch never outlives the index
_INDEX_TTL = 3600  # 1 hour
_LAST_FETCH_TTL = 3600  # 1 hour (same as index)

# Pagination
_PAGE_LIMIT = 500
_MAX_PAGES = 100

# LLM extraction timeout (seconds)
_LLM_TIMEOUT = 30


def _mask_email(email: str) -> str:
    """Mask an email for safe logging: 'alice@example.com' -> 'a***e@example.com'."""
    try:
        local, domain = email.rsplit("@", 1)
        if len(local) <= 2:
            masked_local = local[0] + "***"
        else:
            masked_local = local[0] + "***" + local[-1]
        return f"{masked_local}@{domain}"
    except (ValueError, IndexError):
        return "***"


async def _fetch_tally_page(
    client: Requests,
    form_id: str,
    page: int,
    limit: int = _PAGE_LIMIT,
    start_date: Optional[str] = None,
) -> dict:
    """Fetch a single page of submissions from the Tally API."""
    url = f"{TALLY_API_BASE}/forms/{form_id}/submissions?page={page}&limit={limit}"
    if start_date:
        url += f"&startDate={start_date}"

    response = await client.get(url)
    return response.json()


def _make_tally_client(api_key: str) -> Requests:
    """Create a Requests client configured for the Tally API."""
    return Requests(
        trusted_origins=[TALLY_API_BASE],
        raise_for_status=True,
        extra_headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )


async def _fetch_all_submissions(
    client: Requests,
    form_id: str,
    start_date: Optional[str] = None,
    max_pages: int = _MAX_PAGES,
) -> tuple[list[dict], list[dict]]:
    """Paginate through all Tally submissions. Returns (questions, submissions)."""

    questions: list[dict] = []
    all_submissions: list[dict] = []
    page = 1

    while True:
        data = await _fetch_tally_page(client, form_id, page, start_date=start_date)

        if page == 1:
            questions = data.get("questions", [])

        submissions = data.get("submissions", [])
        all_submissions.extend(submissions)

        # Tally API uses `hasMore` for pagination
        has_more = data.get("hasMore", False)
        if not has_more:
            break
        if page >= max_pages:
            total = data.get("totalNumberOfSubmissionsPerFilter", {}).get("all", "?")
            logger.warning(
                f"Tally: hit max page cap ({max_pages}) for form {form_id}, "
                f"fetched {len(all_submissions)} of {total} total submissions"
            )
            break
        page += 1

    return questions, all_submissions


def _build_email_index(
    submissions: list[dict], questions: list[dict]
) -> dict[str, dict]:
    """Build an {email -> submission_data} index from submissions.

    Scans question titles for email/contact fields to find the email answer.
    """
    # Find question IDs that are likely email fields
    email_question_ids: list[str] = []
    for q in questions:
        label = (q.get("label") or q.get("title") or q.get("name") or "").lower()
        q_type = (q.get("type") or "").lower()
        if q_type in ("input_email", "email"):
            email_question_ids.append(q["id"])
        elif any(kw in label for kw in ("email", "e-mail", "contact")):
            email_question_ids.append(q["id"])

    index: dict[str, dict] = {}
    for sub in submissions:
        email = _extract_email_from_submission(sub, email_question_ids)
        if email:
            index[email.lower()] = {
                "responses": sub.get("responses", sub.get("fields", [])),
                "submitted_at": sub.get("submittedAt", sub.get("createdAt", "")),
                "questions": sub.get("questions", []),
            }
    return index


def _extract_email_from_submission(
    submission: dict, email_question_ids: list[str]
) -> Optional[str]:
    """Extract email address from a submission by checking respondentEmail, then field responses."""
    # Try respondent email first (Tally often includes this)
    respondent_email = submission.get("respondentEmail")
    if respondent_email:
        return respondent_email

    # Search through responses/fields for matching question IDs
    responses = submission.get("responses", submission.get("fields", []))
    if isinstance(responses, list):
        for resp in responses:
            q_id = resp.get("questionId") or resp.get("key") or resp.get("id")
            if q_id in email_question_ids:
                value = resp.get("value") or resp.get("answer")
                if isinstance(value, str) and "@" in value:
                    return value
    elif isinstance(responses, dict):
        for q_id in email_question_ids:
            value = responses.get(q_id)
            if isinstance(value, str) and "@" in value:
                return value

    return None


async def _get_cached_index(
    form_id: str,
) -> tuple[Optional[dict], Optional[list]]:
    """Return (email_index, questions) from Redis, or (None, None) on cache miss."""
    redis = await get_redis_async()
    index_key = _EMAIL_INDEX_KEY.format(form_id=form_id)
    questions_key = _QUESTIONS_KEY.format(form_id=form_id)

    raw_index = await redis.get(index_key)
    raw_questions = await redis.get(questions_key)

    if raw_index and raw_questions:
        return json.loads(raw_index), json.loads(raw_questions)
    return None, None


async def _refresh_cache(form_id: str) -> tuple[dict, list]:
    """Refresh the Tally submission cache. Uses incremental fetch when possible.

    Returns (email_index, questions).
    """
    settings = Settings()
    client = _make_tally_client(settings.secrets.tally_api_key)

    redis = await get_redis_async()
    last_fetch_key = _LAST_FETCH_KEY.format(form_id=form_id)
    index_key = _EMAIL_INDEX_KEY.format(form_id=form_id)
    questions_key = _QUESTIONS_KEY.format(form_id=form_id)

    last_fetch = await redis.get(last_fetch_key)

    if last_fetch:
        # Try to load existing index for incremental merge
        raw_existing = await redis.get(index_key)

        if raw_existing is None:
            # Index expired but last_fetch still present — fall back to full fetch
            logger.info("Tally: last_fetch present but index missing, doing full fetch")
            questions, submissions = await _fetch_all_submissions(client, form_id)
            email_index = _build_email_index(submissions, questions)
        else:
            # Incremental fetch: only get new submissions since last fetch
            logger.info(f"Tally incremental fetch since {last_fetch}")
            questions, new_submissions = await _fetch_all_submissions(
                client, form_id, start_date=last_fetch
            )

            existing_index: dict[str, dict] = json.loads(raw_existing)

            if not questions:
                raw_q = await redis.get(questions_key)
                if raw_q:
                    questions = json.loads(raw_q)

            new_index = _build_email_index(new_submissions, questions)
            existing_index.update(new_index)
            email_index = existing_index
    else:
        # Full initial fetch
        logger.info("Tally full initial fetch")
        questions, submissions = await _fetch_all_submissions(client, form_id)
        email_index = _build_email_index(submissions, questions)

    # Store in Redis
    now = datetime.now(timezone.utc).isoformat()
    await redis.setex(index_key, _INDEX_TTL, json.dumps(email_index))
    await redis.setex(questions_key, _INDEX_TTL, json.dumps(questions))
    await redis.setex(last_fetch_key, _LAST_FETCH_TTL, now)

    logger.info(f"Tally cache refreshed: {len(email_index)} emails indexed")
    return email_index, questions


async def find_submission_by_email(
    form_id: str, email: str
) -> Optional[tuple[dict, list]]:
    """Look up a Tally submission by email. Uses cache when available.

    Returns (submission_data, questions) or None.
    """
    email_lower = email.lower()

    # Try cache first
    email_index, questions = await _get_cached_index(form_id)
    if email_index is not None and questions is not None:
        sub = email_index.get(email_lower)
        if sub is not None:
            return sub, questions
        return None

    # Cache miss - refresh
    email_index, questions = await _refresh_cache(form_id)
    sub = email_index.get(email_lower)
    if sub is not None:
        return sub, questions
    return None


def format_submission_for_llm(submission: dict, questions: list[dict]) -> str:
    """Format a submission as readable Q&A text for LLM consumption."""
    # Build question ID -> title lookup
    q_titles: dict[str, str] = {}
    for q in questions:
        q_id = q.get("id", "")
        title = q.get("label") or q.get("title") or q.get("name") or f"Question {q_id}"
        q_titles[q_id] = title

    lines: list[str] = []
    responses = submission.get("responses", [])

    if isinstance(responses, list):
        for resp in responses:
            q_id = resp.get("questionId") or resp.get("key") or resp.get("id") or ""
            title = q_titles.get(q_id, f"Question {q_id}")
            value = resp.get("value") or resp.get("answer") or ""
            lines.append(f"Q: {title}\nA: {_format_answer(value)}")
    elif isinstance(responses, dict):
        for q_id, value in responses.items():
            title = q_titles.get(q_id, f"Question {q_id}")
            lines.append(f"Q: {title}\nA: {_format_answer(value)}")

    return "\n\n".join(lines)


def _format_answer(value: object) -> str:
    """Convert an answer value (str, list, dict, None) to a human-readable string."""
    if value is None:
        return "(no answer)"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items() if v]
        return "; ".join(parts) if parts else "(no answer)"
    return str(value)


_EXTRACTION_PROMPT = """\
You are a business analyst. Given the following form submission data, extract structured business understanding information.

Return a JSON object with ONLY the fields that can be confidently extracted. Use null for fields that cannot be determined.

Fields:
- user_name (string): the person's name
- job_title (string): their job title
- business_name (string): company/business name
- industry (string): industry or sector
- business_size (string): company size e.g. "1-10", "11-50", "51-200"
- user_role (string): their role context e.g. "decision maker", "implementer"
- key_workflows (list of strings): key business workflows
- daily_activities (list of strings): daily activities performed
- pain_points (list of strings): current pain points
- bottlenecks (list of strings): process bottlenecks
- manual_tasks (list of strings): manual/repetitive tasks
- automation_goals (list of strings): desired automation goals
- current_software (list of strings): software/tools currently used
- existing_automation (list of strings): existing automations
- additional_notes (string): any additional context

Form data:
"""

_EXTRACTION_SUFFIX = "\n\nReturn ONLY valid JSON."


async def extract_business_understanding(
    formatted_text: str,
) -> BusinessUnderstandingInput:
    """Use an LLM to extract structured business understanding from form text.

    Raises on timeout or unparseable response so the caller can handle it.
    """
    settings = Settings()
    api_key = settings.secrets.open_router_api_key
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"{_EXTRACTION_PROMPT}{formatted_text}{_EXTRACTION_SUFFIX}",
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Tally: LLM extraction timed out")
        raise

    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Tally: LLM returned invalid JSON, skipping extraction")
        raise

    # Filter out null values before constructing
    cleaned = {k: v for k, v in data.items() if v is not None}
    return BusinessUnderstandingInput(**cleaned)


async def populate_understanding_from_tally(user_id: str, email: str) -> None:
    """Main orchestrator: check Tally for a matching submission and populate understanding.

    Fire-and-forget safe — all exceptions are caught and logged.
    """
    try:
        # Check if understanding already exists (idempotency)
        existing = await get_business_understanding(user_id)
        if existing is not None:
            logger.debug(
                f"Tally: user {user_id} already has business understanding, skipping"
            )
            return

        # Check API key is configured
        settings = Settings()
        if not settings.secrets.tally_api_key:
            logger.debug("Tally: no API key configured, skipping")
            return

        # Look up submission by email
        masked = _mask_email(email)
        result = await find_submission_by_email(TALLY_FORM_ID, email)
        if result is None:
            logger.debug(f"Tally: no submission found for {masked}")
            return

        submission, questions = result
        logger.info(f"Tally: found submission for {masked}, extracting understanding")

        # Format and extract
        formatted = format_submission_for_llm(submission, questions)
        if not formatted.strip():
            logger.warning("Tally: formatted submission was empty, skipping")
            return

        understanding_input = await extract_business_understanding(formatted)

        # Upsert into database
        await upsert_business_understanding(user_id, understanding_input)
        logger.info(f"Tally: successfully populated understanding for user {user_id}")

    except Exception:
        logger.exception(f"Tally: error populating understanding for user {user_id}")
