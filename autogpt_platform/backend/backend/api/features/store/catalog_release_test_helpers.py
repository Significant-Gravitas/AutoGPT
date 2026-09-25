"""Fixtures for the disposable PostgreSQL catalogue publisher integration test."""

import hashlib

from backend.api.features.store.catalog_release_load import LoadedPackage, LoadedRelease
from backend.api.features.store.catalog_release_model import (
    Adoption,
    ReleaseManifest,
    digest,
)
from backend.copilot.tools.skills import SkillFile
from backend.data import db as database


def release(
    number: int, *, second: bool = False, retire_first: bool = False
) -> LoadedRelease:
    slugs = ["catalog-test-second"] if retire_first else ["catalog-test-demo"]
    if second and "catalog-test-second" not in slugs:
        slugs.append("catalog-test-second")
    slugs.append("catalog-test-baseline-extra")
    slugs.append("catalog-test-baseline-unversioned")
    packages = {slug: package(slug, number) for slug in slugs}
    manifest = ReleaseManifest.model_validate(
        {
            "schema_version": 1,
            "release_key": f"test-{number}",
            "catalog_sha256": "a" * 64,
            "packages": [
                {"slug": slug, "tree_sha256": digest(files), "files": files}
                for slug, files in [
                    (
                        slug,
                        [
                            {
                                "path": "SKILL.md",
                                "sha256": hashlib.sha256(
                                    p.skill_markdown.encode()
                                ).hexdigest(),
                                "executable": False,
                            },
                            {
                                "path": "scripts/run.py",
                                "sha256": hashlib.sha256(
                                    p.files[0].content
                                ).hexdigest(),
                                "executable": True,
                            },
                        ],
                    )
                    for slug, p in packages.items()
                ]
            ],
            "experts": [{"key": "max", "skills": slugs}],
            "retirements": ["catalog-test-demo"] if retire_first else [],
        }
    )
    return LoadedRelease(
        revision=f"{number:040x}",
        release_id=digest({"test": number}),
        manifest_sha256=digest(manifest.model_dump()),
        manifest=manifest,
        packages=packages,
    )


def package(slug: str, number: int) -> LoadedPackage:
    markdown = f"---\nname: {slug}\ndescription: Test package\nmetadata:\n  custom: preserved\n---\n\nBody {number}\n"
    return LoadedPackage(
        slug=slug,
        package_sha256=digest({"slug": slug, "number": number}),
        skill_markdown=markdown,
        name=slug,
        description="Test package",
        body=f"Body {number}\n",
        triggers=[],
        categories=["content"],
        required_providers=[],
        source_repo=None,
        source_url=None,
        license="MIT",
        files=[
            SkillFile(
                relative_path="scripts/run.py",
                content=f"print({number})\n".encode(),
                is_executable=True,
            )
        ],
    )


async def seed_personal_and_platform_records() -> Adoption:
    sql = [
        """INSERT INTO "User" (id, email, "updatedAt") VALUES ('catalog-user', 'catalog-test@example.invalid', now())""",
        """INSERT INTO "Profile" (id,"userId",name,username,description,links,"updatedAt")
        VALUES ('catalog-profile','catalog-user','Private','catalog-test','Private',ARRAY[]::text[],now())""",
        """INSERT INTO "Organization" (id,name,slug,"updatedAt") VALUES ('catalog-org','Private','catalog-test-org',now())""",
        """INSERT INTO "Expert" (id,name,role,identity,"isTemplate","isArchived")
        VALUES ('catalog-template','Max','Test','Template',true,true)""",
        """INSERT INTO "Expert" (id,name,role,identity,"ownerUserId","sourceTemplateId") VALUES
        ('catalog-custom','Private','Custom','User authored','catalog-user',NULL),
        ('catalog-hire','Max','Hired','User customised','catalog-user','catalog-template')""",
        """INSERT INTO "SkillListing" (id,slug,"owningUserId","owningOrgId") VALUES
        ('catalog-private-skill','private-skill','catalog-user',NULL),
        ('catalog-org-skill','org-skill',NULL,'catalog-org'),
        ('catalog-old-skill','catalog-test-demo',NULL,NULL),
        ('catalog-unversioned','catalog-test-baseline-unversioned',NULL,NULL),
        ('catalog-unmanaged','unmanaged-platform-skill',NULL,NULL)""",
        """INSERT INTO "SkillListingVersion" (id,version,name,description,body,triggers,categories,"skillListingId") VALUES
        ('catalog-private-version',1,'private-skill','Mine','Private body',ARRAY[]::text[],ARRAY['content'],'catalog-private-skill'),
        ('catalog-org-version',1,'org-skill','Ours','Org body',ARRAY[]::text[],ARRAY['content'],'catalog-org-skill'),
        ('catalog-old-version',1,'catalog-test-demo','Old','Old body',ARRAY[]::text[],ARRAY['content'],'catalog-old-skill'),
        ('catalog-unmanaged-version',1,'unmanaged-platform-skill','Unmanaged','Original',ARRAY[]::text[],ARRAY['content'],'catalog-unmanaged')""",
        """UPDATE "SkillListing" SET "activeVersionId" = CASE id
        WHEN 'catalog-private-skill' THEN 'catalog-private-version'
        WHEN 'catalog-org-skill' THEN 'catalog-org-version'
        WHEN 'catalog-old-skill' THEN 'catalog-old-version'
        ELSE 'catalog-unmanaged-version' END, "hasApprovedVersion" = true
        WHERE id IN ('catalog-private-skill','catalog-org-skill','catalog-old-skill','catalog-unmanaged')""",
        """INSERT INTO "SkillListingFile" (id,"skillListingVersionId","relativePath","sizeBytes",sha256,content)
        VALUES ('catalog-private-file','catalog-private-version','notes.txt',7,'private',convert_to('Private','UTF8')),
        ('catalog-unmanaged-file','catalog-unmanaged-version','notes.txt',7,'original',convert_to('Original','UTF8'))""",
        """INSERT INTO "ExpertSkillListing" ("expertId","skillListingId") VALUES
        ('catalog-template','catalog-old-skill'), ('catalog-custom','catalog-private-skill')""",
        """INSERT INTO "UserWorkspace" (id,"userId","updatedAt") VALUES ('catalog-workspace','catalog-user',now())""",
        """INSERT INTO "UserWorkspaceFile" (id,"workspaceId",name,path,"storagePath","mimeType","sizeBytes","updatedAt",metadata)
        VALUES ('catalog-workspace-file','catalog-workspace','SKILL.md','/skills/catalog-test-demo/SKILL.md','private/blob','text/markdown',123,now(),'{"origin":"user"}')""",
        """INSERT INTO "ExpertRoutine" (id,"expertId",title,prompt,crons,asks,"scheduleIds")
        VALUES ('catalog-routine','catalog-hire','Private routine','User prompt',ARRAY['0 9 * * *'],ARRAY[]::text[],ARRAY['private-schedule'])""",
        """INSERT INTO "ExpertCredential" (id,"expertId","credentialId",provider)
        VALUES ('catalog-credential','catalog-hire','private-credential','test')""",
    ]
    async with database.transaction() as tx:
        for statement in sql:
            await tx.execute_raw(statement)
    return Adoption(
        skills={
            "catalog-test-demo": "catalog-old-skill",
            "catalog-test-baseline-unversioned": "catalog-unversioned",
        },
        experts={"max": "catalog-template"},
        activate_experts=["max"],
    )


async def protected_snapshot() -> list:
    return await database.prisma.query_raw(
        """SELECT jsonb_build_object(
        'skills', (SELECT jsonb_agg(to_jsonb(s) ORDER BY id) FROM "SkillListing" s WHERE "owningUserId" IS NOT NULL OR "owningOrgId" IS NOT NULL),
        'versions', (SELECT jsonb_agg(to_jsonb(v) ORDER BY id) FROM "SkillListingVersion" v WHERE id IN ('catalog-private-version','catalog-org-version')),
        'files', (SELECT jsonb_agg(to_jsonb(f) ORDER BY id) FROM "SkillListingFile" f WHERE id='catalog-private-file'),
        'experts', (SELECT jsonb_agg(to_jsonb(e) ORDER BY id) FROM "Expert" e WHERE "ownerUserId" IS NOT NULL),
        'workspace', (SELECT jsonb_agg(to_jsonb(w) ORDER BY id) FROM "UserWorkspaceFile" w),
        'routines', (SELECT jsonb_agg(to_jsonb(r) ORDER BY id) FROM "ExpertRoutine" r),
        'credentials', (SELECT jsonb_agg(to_jsonb(c) ORDER BY id) FROM "ExpertCredential" c)
        )::text AS snapshot"""
    )
