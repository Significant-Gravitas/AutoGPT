# Publishing marketplace catalogue releases

The private `Significant-Gravitas/skills-catalog` repository owns marketplace
skill packages, ordered expert assignments and explicit retirements. The backend
validates and publishes an exact catalogue commit into the existing database.
Runtime reads use that published database state; they do not fetch GitHub.

This migration has two distinct stages: adopt the preserved existing catalogue,
then publish content replacements through the same mechanism. Removing a package
from a checkout is never authority to delete or retire a database record.

## Scope and ownership

Every operation requires an environment-specific adoption JSON file. It names
exact platform listing and template IDs. Matching a name does not grant ownership.
Listings must have both owner columns null. Templates must have `isTemplate=true`
and no user, organisation or team owner. A collision or unexpected drift aborts.

The publisher and rollback write only those marketplace parents and their
versions, package files and assignments. They do not call the full expert seed,
customer setup, workspace installation, routine backfill or credential helpers.
Existing users retain their installed copies. A later user-requested install is
a separate operation and retains the existing protection against overwriting a
user-authored skill folder.

After adoption, database triggers protect catalogue-owned pointers, assignments
and immutable versions against the obsolete writing paths. Install counts and
virus-scan caches remain writable. The guard functions use a fixed search path
and owner privileges so row-level security cannot hide catalogue ownership from
the guard. These controls do not constrain a database administrator who can alter
the schema itself.

## Release and installation behavior

`release.json` binds `catalog.yml`, every package file, Git executable modes and
ordered expert assignments. The loader requires a clean checkout at the supplied
40-character commit and checks bytes against both the manifest and Git objects.
Full catalogue entry metadata is retained in `catalogueMetadata` and participates
in immutable package identity. Existing SKILL.md frontmatter remains the source
of displayed attribution; importing does not silently replace it with catalogue
metadata. Raw SKILL.md bytes, additional metadata, companion files and executable
flags are retained for installation.

Changed packages get new immutable versions. Activation switches the package
versions, assignments, explicit retirements and release record in one transaction.
Repeating an already active release is a validated no-op. Concurrent activations
serialize; a stale preview fails instead of overwriting a newer release.

New hires record their exact package versions while sharing the publication
lock. Setup retries use those versions even after the marketplace changes.
Completed hires are not upgraded by publication. An old incomplete hire with no
saved package snapshot stops before setup writes: recover its original bundle
explicitly; do not fill it from today's template or bulk-rewrite customer rows.

Preserving metadata is not proof that every imported skill's external services,
tools, scripts or runtime-specific instructions work. Test those requirements in
the development runtime before promoting a content release. Nonempty
`system_packages` are rejected until a supported system-package loader exists.

## First adoption: operational prerequisites

Do these checks separately for development and production:

1. Deploy the schema and compatible backend. Verify the actual running images and
   source in every service/job that can write skills, templates or hire setup;
   a requested Git revision or image tag alone is insufficient evidence.
2. Disable obsolete seeding jobs and pause/drain relevant older requests and
   background writers. Account for old pods, pending jobs and scheduled writers.
   Existing transactions that passed the old unadopted guard can otherwise commit
   after activation. Database triggers do not eliminate this first-cutover step.
3. Inventory incomplete legacy hires and resolve or isolate them without guessing
   their original skill assignments. Confirm new hires/retries use saved versions.
4. Capture fresh scoped marketplace backups and ownership inventory. Record the
   database host/schema, backend image, catalogue commit and exact adoption files.
   Do not copy users, credentials or entire databases between environments.
5. Provision missing templates privately, preview the baseline against the final
   state, inspect the proposed scope, and apply. Verify unchanged personal records
   while distinguishing normal concurrent user activity from migration writes.

The automatic routine publisher must refuse an unadopted environment. First
adoption is an explicit cutover, not a side effect of enabling a scheduled job.
Feature flags have not been established as a substitute for the required drain.

## Create missing platform templates

`catalogue_templates` requires every roster key mapped to an exact existing
template ID or `null` for an explicitly expected-absent template:

```json
{"schema_version": 1, "experts": {"example": "existing-template-id"}}
```

Run from the backend directory with the intended environment configured:

```sh
poetry run python -m backend.api.features.experts.catalogue_templates preview \
  --adoption templates.json --output templates-preview.json
poetry run python -m backend.api.features.experts.catalogue_templates apply \
  --adoption templates.json --preview templates-preview.json \
  --output provisioned-templates.json
```

The result contains `experts` and `activate_experts` for the release adoption
file. Newly created templates remain archived until their catalogue release
activates them. Routines start disabled and no credentials are granted. Existing
templates, routines, preloads, credentials and customer copies are unchanged.
Preloads must resolve to active approved listings of the official `autogpt`
profile. Provisioning refuses missing, ambiguous or stale dependencies.

This command does not apply wording changes to existing templates. Such changes
need a separately listed, scoped template update. Do not use the retired full
seeder as a substitute.

## Preview and publish

An adoption file maps every active, retired and previously managed skill, plus
every expert in the manifest. `null` means the listing must not exist. It is not
a wildcard. After creating listings, record their actual IDs for future releases.

```json
{
  "skills": {"example-skill": "existing-listing-id", "new-skill": null},
  "experts": {"example": "existing-template-id"},
  "activate_experts": []
}
```

The backend runtime image includes Git and tar for validated catalogue transfer.
Verify both in the built image before enabling a pod-based publishing job.
Use Git on the publishing runner and a narrowly scoped read credential to prepare
a clean checkout. Keep checkout credentials out of artifacts and output. Never
substitute a branch name for `CATALOGUE_REVISION` below.

```sh
poetry run python -m backend.api.features.store.catalog_release preview \
  --catalogue /path/to/clean/catalogue --revision "$CATALOGUE_REVISION" \
  --adoption adoption.json --output preview.json
poetry run python -m backend.api.features.store.catalog_release apply \
  --catalogue /path/to/clean/catalogue --revision "$CATALOGUE_REVISION" \
  --adoption adoption.json --approved preview.json --output applied.json
```

Approval is bound to the database target, exact release, adoption scope and current
managed state. Review a fresh preview after drift; do not edit a stale preview to
force it through. `update_skills` lists existing packages processed by the release,
including packages whose matching immutable version is reused; it is not a count
of changed file contents. Publish and exercise a release in development first. Production
promotion must consume the same validated catalogue revision and compatible
backend revision, with a retained successful development verification record.

## Rollback and evidence

Use a recorded release ID and the current complete adoption map:

```sh
poetry run python -m backend.api.features.store.catalog_release preview-rollback \
  --release-id "$PREVIOUS_RELEASE_ID" --adoption adoption.json \
  --output rollback-preview.json
poetry run python -m backend.api.features.store.catalog_release rollback \
  --release-id "$PREVIOUS_RELEASE_ID" --adoption adoption.json \
  --approved rollback-preview.json --output rolled-back.json
```

The first publication also records an immutable `database-before-adoption`
snapshot in the same transaction. Its ID is exposed as `rollback_release_id` in
the preview. That snapshot is explicitly a database recovery point, not a Git
catalogue release. It preserves original version pointers, approval/visibility
flags, template visibility and ordered assignments. Listings created by adoption
return to hidden and unapproved, with no active version, when that snapshot is
restored.

Rollback retains versions and newly created listing rows; it restores marketplace
visibility and assignments without reverting user copies or unrelated counters.
Rollback across a changed expert key set requires a separate reviewed template
migration and fails closed. Rolling the application back to code with obsolete
writers also requires a compatibility check.

Retain the before/after marketplace backup, preview, apply/rollback result,
catalogue revision, actual backend image and verification results. The dedicated
`Catalogue safety` CI job uses an empty local PostgreSQL database, applies the
actual migration and tests ownership, package bytes, retries, concurrent writes,
failure atomicity and rollback. It fails if safety tests are skipped. These tests
do not replace development checks of actual sandbox files, executable scripts,
new hires, retried setup and unchanged existing user copies.

## Legacy commands and remaining extraction

`backend.api.features.store.skill_seed` and
`backend.api.features.experts.seed.seed_roster` now fail clearly before writing.
The bundled marketplace content and Python assignment/retirement lists are
removed. CI rejects their reintroduction. Persona/style evaluation still reads
expert definitions; its rendered prompts do not consume the old skill list.

The internal `agent_building_guide` still reads its application instruction file
from AutoGPT. Its catalogue-backed system-package migration is separate work;
do not describe this initial integration as removing all application skills.
