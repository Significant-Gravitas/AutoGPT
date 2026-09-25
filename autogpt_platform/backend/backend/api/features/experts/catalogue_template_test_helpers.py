"""Scoped data builders for the dedicated local template-provisioning tests."""

import copy
from uuid import uuid4

from backend.api.features.experts import catalogue_templates as provisioning


class Fixture:
    def __init__(self, db):
        self.db = db
        self.experts = []
        self.users = []
        self.listings = []
        self.graphs = []
        self.names = []

    async def user(self):
        record = await self.db.user.create(
            data={"id": str(uuid4()), "email": f"{uuid4()}@example.invalid"}
        )
        self.users.append(record.id)
        return record.id

    async def expert(self, name, **kwargs):
        record = await self.db.expert.create(
            data={
                "name": name,
                "role": "Original role",
                "identity": "Owner's original text",
                **kwargs,
            }
        )
        self.experts.append(record.id)
        return record

    async def workflow(self, slug, *, official):
        username = "autogpt" if official else f"test-{uuid4()}"
        profile = await self.db.profile.find_unique(where={"username": username})
        if profile is None:
            owner = await self.user()
            profile = await self.db.profile.create(
                data={
                    "userId": owner,
                    "name": "Test creator",
                    "username": username,
                    "description": "Test only",
                    "links": [],
                }
            )
        graph = await self.db.agentgraph.create(data={"userId": profile.userId})
        self.graphs.append(graph.id)
        listing = await self.db.storelisting.create(
            data={
                "slug": slug,
                "owningUserId": profile.userId,
                "agentGraphId": graph.id,
                "hasApprovedVersion": True,
            }
        )
        self.listings.append(listing.id)
        version = await self.db.storelistingversion.create(
            data={
                "storeListingId": listing.id,
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
                "name": "Test workflow",
                "subHeading": "Test",
                "description": "Test",
                "imageUrls": [],
                "categories": ["operations"],
                "submissionStatus": "APPROVED",
            }
        )
        await self.db.storelisting.update(
            where={"id": listing.id}, data={"activeVersionId": version.id}
        )
        return version.id

    def definitions(self, monkeypatch, count=1):
        result = {}
        for _ in range(count):
            key = f"test-{uuid4().hex}"
            name = key.title()
            self.names.append(name)
            result[key] = {
                "fields": {
                    "name": name,
                    "role": "Operations",
                    "jobTitle": "Operations expert",
                    "tagline": "Reviewed template",
                    "avatarUrl": None,
                    "identity": "Reviewed persona",
                    "bio": "Bio",
                    "voicePreferences": "Voice",
                    "boundaries": "Boundaries",
                    "categories": ["operations"],
                    "dayOne": [],
                },
                "routines": [
                    {
                        "key": "weekly-review",
                        "title": "Review",
                        "prompt": "Reviewed proposal",
                        "crons": ["0 9 * * 1"],
                        "asks": ["Which project?"],
                        "sessionMode": "THREAD",
                    }
                ],
                "preloads": [],
            }
        monkeypatch.setattr(
            provisioning,
            "template_definitions",
            lambda: {
                key: provisioning.TemplateDefinition.model_validate(
                    copy.deepcopy(value)
                )
                for key, value in result.items()
            },
        )
        return result

    async def fingerprint(self, ids):
        return await provisioning._template_rows(self.db, ids)

    async def cleanup(self):
        await self.db.expert.delete_many(
            where={"OR": [{"id": {"in": self.experts}}, {"name": {"in": self.names}}]}
        )
        await self.db.storelisting.update_many(
            where={"id": {"in": self.listings}}, data={"activeVersionId": None}
        )
        await self.db.storelisting.delete_many(where={"id": {"in": self.listings}})
        await self.db.agentgraph.delete_many(where={"id": {"in": self.graphs}})
        await self.db.user.delete_many(where={"id": {"in": self.users}})
