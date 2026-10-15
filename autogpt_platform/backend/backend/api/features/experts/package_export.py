"""Build a marketplace template's ``.expert.zip`` (format in ``package_model``).

Every part comes from the path that already serves it: skills are the live Hub
versions ``hire_expert`` installs, and each workflow file is the body
``download_agent_file`` returns for its listing, so it also imports alone.
"""

import asyncio
import io
import logging
import zipfile

import backend.util.json
from backend.api.features.experts import experts_db, routines
from backend.api.features.experts.models import (
    Expert,
    ExpertBundledSkill,
    ExpertWorkflowRef,
)
from backend.api.features.experts.package_model import (
    MANIFEST_PATH,
    ExpertManifest,
    ExpertPackage,
    PackagedDayOneItem,
    PackagedRoutine,
    PackagedSkill,
    PackagedVoiceSample,
    PackagedWorkflow,
    manifest_json,
    skill_path,
    validate_expert_package,
    workflow_path,
)
from backend.api.features.skill_zip import add_entry, write_skill_package
from backend.api.features.store import db as store_db
from backend.api.features.store import skill_db
from backend.copilot.tools.skills import SkillPackage
from backend.data.graph import GraphModel
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


async def build_template_package(template_id: str, user_id: str) -> ExpertPackage:
    """The live template *template_id* as a package; ``NotFoundError`` when it
    is missing or archived."""
    template = await experts_db.get_template(template_id)
    if template is None:
        raise NotFoundError(f"Expert template #{template_id} not found")
    [bundled] = await experts_db.with_bundled_skills([template], user_id)
    (skill_cards, skills), (workflow_entries, workflows), template_routines = (
        await asyncio.gather(
            _skills(bundled.bundled_skills),
            _workflows(template.workflows),
            routines.list_template_routines(template_id),
        )
    )
    manifest = _manifest(
        template,
        skill_cards,
        workflow_entries,
        [
            PackagedRoutine(
                key=r.key,
                title=r.title,
                prompt=r.prompt,
                crons=r.crons,
                asks=r.asks,
                session_mode=r.session_mode,
            )
            for r in template_routines
        ],
    )
    return ExpertPackage(manifest=manifest, skills=skills, workflows=workflows)


def zip_from_expert_package(package: ExpertPackage) -> bytes:
    """Write the zip: ``expert.json``, then exactly the skills and workflows
    the manifest names, in its order. CPU-bound; run it off the event loop."""
    validate_expert_package(package)
    manifest = package.manifest
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        add_entry(archive, MANIFEST_PATH, manifest_json(manifest))
        for card in manifest.skills:
            write_skill_package(archive, package.skills[card.slug], prefix=card.path)
        for entry in manifest.workflows:
            add_entry(archive, entry.path, package.workflows[entry.path])
    return buffer.getvalue()


def _manifest(
    template: Expert,
    skills: list[PackagedSkill],
    workflows: list[PackagedWorkflow],
    template_routines: list[PackagedRoutine],
) -> ExpertManifest:
    return ExpertManifest(
        name=template.name,
        role=template.role,
        job_title=template.job_title,
        tagline=template.tagline,
        bio=template.bio,
        categories=template.categories,
        color=template.color,
        avatar_url=template.avatar_url,
        identity=template.identity,
        voice_preferences=template.voice_preferences,
        voice_samples=[
            PackagedVoiceSample.model_validate(s.model_dump())
            for s in template.voice_samples
        ],
        boundaries=template.boundaries,
        day_one=[
            PackagedDayOneItem.model_validate(d.model_dump()) for d in template.day_one
        ],
        skills=skills,
        workflows=workflows,
        routines=template_routines,
    )


async def _skills(
    bundled: list[ExpertBundledSkill],
) -> tuple[list[PackagedSkill], dict[str, SkillPackage]]:
    """Each bundled skill's active version, whole, in roster order."""
    active = await skill_db.get_active_versions([s.slug for s in bundled])
    version_ids = {
        slug: version.version_id
        for slug, version in active.items()
        if not version.retired
    }
    versions = await skill_db.get_version_packages(list(version_ids.values()))
    cards: list[PackagedSkill] = []
    packages: dict[str, SkillPackage] = {}
    for skill in bundled:
        version = versions.get(version_ids.get(skill.slug, ""))
        if version is None:
            logger.warning(f"Bundled skill {skill.slug!r} has no live version")
            continue
        cards.append(
            PackagedSkill(
                slug=skill.slug,
                title=skill.title,
                description=skill.description,
                path=skill_path(skill.slug),
            )
        )
        packages[skill.slug] = SkillPackage(
            skill_md=version.skill_markdown, files=version.files
        )
    return cards, packages


async def _workflows(
    refs: list[ExpertWorkflowRef],
) -> tuple[list[PackagedWorkflow], dict[str, bytes]]:
    """Each workflow's agent file, numbered in roster order."""
    listed = [
        (ref, listing_id)
        for ref in refs
        if (listing_id := ref.store_listing_version_id)
    ]
    graphs = await asyncio.gather(
        *(_agent_file(listing_id) for _, listing_id in listed)
    )
    entries: list[PackagedWorkflow] = []
    files: dict[str, bytes] = {}
    for (ref, listing_id), graph in zip(listed, graphs):
        if graph is None:
            continue
        path = workflow_path(len(entries), ref.name or graph.name)
        entries.append(
            PackagedWorkflow(
                path=path,
                name=ref.name,
                description=ref.description,
                store_listing_version_id=listing_id,
                schedule_cron=ref.schedule_cron,
                integration_providers=ref.integration_providers,
            )
        )
        files[path] = backend.util.json.dumps(graph).encode("utf-8")
    return entries, files


async def _agent_file(store_listing_version_id: str) -> GraphModel | None:
    """The graph ``download_agent_file`` serves, or ``None`` for a listing no
    longer installable: there is no graph to put in the file."""
    try:
        return await store_db.get_agent(store_listing_version_id)
    except NotFoundError:
        logger.warning(
            f"Template workflow listing {store_listing_version_id} is no longer "
            "installable; left out of the expert file"
        )
        return None
