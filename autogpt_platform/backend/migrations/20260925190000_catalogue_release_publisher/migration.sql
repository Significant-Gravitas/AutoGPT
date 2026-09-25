ALTER TABLE "Expert" ADD COLUMN "skillInstallSnapshot" JSONB;
ALTER TABLE "SkillListingVersion" ADD COLUMN "skillMarkdown" TEXT;
ALTER TABLE "SkillListingVersion" ADD COLUMN "cataloguePackageSha256" TEXT;
ALTER TABLE "SkillListingVersion" ADD COLUMN "catalogueMetadata" JSONB;
CREATE UNIQUE INDEX "SkillListingVersion_skillListingId_cataloguePackageSha256_key"
ON "SkillListingVersion"("skillListingId", "cataloguePackageSha256");

CREATE TABLE "CatalogueRelease" (
    id TEXT PRIMARY KEY, revision TEXT NOT NULL, "manifestSha256" TEXT NOT NULL,
    manifest JSONB NOT NULL, snapshot JSONB NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE "CatalogueReleaseVersion" (
    "releaseId" TEXT NOT NULL REFERENCES "CatalogueRelease"(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    "versionId" TEXT NOT NULL REFERENCES "SkillListingVersion"(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    PRIMARY KEY ("releaseId", "versionId")
);
CREATE INDEX "CatalogueReleaseVersion_versionId_idx" ON "CatalogueReleaseVersion"("versionId");
CREATE TABLE "CatalogueState" (
    id TEXT PRIMARY KEY CHECK (id = 'marketplace'),
    "activeReleaseId" TEXT REFERENCES "CatalogueRelease"(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    generation INTEGER NOT NULL DEFAULT 0, snapshot JSONB,
    CHECK (("activeReleaseId" IS NULL) = (snapshot IS NULL))
);
INSERT INTO "CatalogueState"(id) VALUES ('marketplace');
CREATE TABLE "CatalogueActivation" (
    id TEXT PRIMARY KEY,
    "releaseId" TEXT NOT NULL REFERENCES "CatalogueRelease"(id) ON DELETE RESTRICT ON UPDATE CASCADE,
    "previousReleaseId" TEXT,
    generation INTEGER NOT NULL UNIQUE, rollback BOOLEAN NOT NULL DEFAULT false,
    snapshot JSONB NOT NULL, "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "CatalogueRelease" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "CatalogueReleaseVersion" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "CatalogueState" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "CatalogueActivation" ENABLE ROW LEVEL SECURITY;

CREATE FUNCTION catalogue_managed(kind TEXT, record_id TEXT) RETURNS BOOLEAN
LANGUAGE SQL STABLE AS $$
    SELECT EXISTS (SELECT 1 FROM "CatalogueState" s,
        jsonb_each(s.snapshot -> kind) item
        WHERE item.value ->> CASE WHEN kind = 'skills' THEN 'listing_id' ELSE 'expert_id' END = record_id)
$$;

CREATE FUNCTION catalogue_publisher_enabled() RETURNS BOOLEAN LANGUAGE SQL STABLE AS $$
    SELECT COALESCE(current_setting('autogpt.catalogue_publisher', true), '') = 'on'
$$;

CREATE FUNCTION guard_catalogue_parent() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NOT catalogue_managed(CASE WHEN TG_TABLE_NAME = 'SkillListing' THEN 'skills' ELSE 'experts' END, OLD.id) THEN
        IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
    END IF;
    IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'managed catalogue records cannot be deleted'; END IF;
    IF TG_TABLE_NAME = 'SkillListing' AND
        jsonb_build_array(to_jsonb(NEW)->'id', to_jsonb(NEW)->'owningUserId', to_jsonb(NEW)->'owningOrgId')
        IS DISTINCT FROM jsonb_build_array(to_jsonb(OLD)->'id', to_jsonb(OLD)->'owningUserId', to_jsonb(OLD)->'owningOrgId') THEN
        RAISE EXCEPTION 'managed skill ownership is immutable';
    END IF;
    IF TG_TABLE_NAME = 'SkillListing' AND
        (to_jsonb(NEW) - 'updatedAt' - 'installCount') IS DISTINCT FROM (to_jsonb(OLD) - 'updatedAt' - 'installCount')
        AND NOT catalogue_publisher_enabled() THEN
        RAISE EXCEPTION 'managed skill changes require the catalogue publisher';
    END IF;
    IF TG_TABLE_NAME = 'Expert' AND
        jsonb_build_array(to_jsonb(NEW)->'id', to_jsonb(NEW)->'ownerUserId', to_jsonb(NEW)->'organizationId', to_jsonb(NEW)->'teamId', to_jsonb(NEW)->'isTemplate')
        IS DISTINCT FROM jsonb_build_array(to_jsonb(OLD)->'id', to_jsonb(OLD)->'ownerUserId', to_jsonb(OLD)->'organizationId', to_jsonb(OLD)->'teamId', to_jsonb(OLD)->'isTemplate') THEN
        RAISE EXCEPTION 'managed expert ownership is immutable';
    END IF;
    IF TG_TABLE_NAME = 'Expert' AND to_jsonb(NEW)->'isArchived' IS DISTINCT FROM to_jsonb(OLD)->'isArchived'
        AND NOT catalogue_publisher_enabled() THEN
        RAISE EXCEPTION 'managed expert visibility requires the catalogue publisher';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER catalogue_skill_guard BEFORE UPDATE OR DELETE ON "SkillListing"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_parent();
CREATE TRIGGER catalogue_expert_guard BEFORE UPDATE OR DELETE ON "Expert"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_parent();

CREATE FUNCTION guard_catalogue_assignment() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NOT catalogue_publisher_enabled() AND (
        (TG_OP <> 'INSERT' AND catalogue_managed('experts', OLD."expertId")) OR
        (TG_OP <> 'DELETE' AND catalogue_managed('experts', NEW."expertId"))) THEN
        RAISE EXCEPTION 'managed expert assignments require the catalogue publisher';
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END $$;
CREATE TRIGGER catalogue_assignment_guard BEFORE INSERT OR UPDATE OR DELETE ON "ExpertSkillListing"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_assignment();

CREATE FUNCTION guard_catalogue_version() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE protected BOOLEAN;
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF catalogue_managed('skills', NEW."skillListingId") AND NOT catalogue_publisher_enabled() THEN
            RAISE EXCEPTION 'managed skill versions require the catalogue publisher';
        END IF;
        RETURN NEW;
    END IF;
    protected := catalogue_managed('skills', OLD."skillListingId") OR
        EXISTS (SELECT 1 FROM "CatalogueReleaseVersion" WHERE "versionId" = OLD.id);
    IF TG_OP = 'UPDATE' THEN
        protected := protected OR catalogue_managed('skills', NEW."skillListingId") OR
            EXISTS (SELECT 1 FROM "CatalogueReleaseVersion" WHERE "versionId" = NEW.id);
    END IF;
    IF protected THEN
        IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'catalogue versions are immutable'; END IF;
        IF (to_jsonb(NEW) - 'scannedSha256' - 'updatedAt') IS DISTINCT FROM
            (to_jsonb(OLD) - 'scannedSha256' - 'updatedAt') THEN
            RAISE EXCEPTION 'catalogue versions are immutable';
        END IF;
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END $$;
CREATE TRIGGER catalogue_version_guard BEFORE INSERT OR UPDATE OR DELETE ON "SkillListingVersion"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_version();

CREATE FUNCTION guard_catalogue_file() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE version_id TEXT; parent_id TEXT; new_parent_id TEXT;
BEGIN
    version_id := CASE WHEN TG_OP = 'INSERT' THEN NEW."skillListingVersionId" ELSE OLD."skillListingVersionId" END;
    SELECT "skillListingId" INTO parent_id FROM "SkillListingVersion" WHERE id = version_id;
    IF EXISTS (SELECT 1 FROM "CatalogueReleaseVersion" WHERE "versionId" = version_id) OR
        (catalogue_managed('skills', parent_id) AND (TG_OP <> 'INSERT' OR NOT catalogue_publisher_enabled())) THEN
        RAISE EXCEPTION 'catalogue package files are immutable';
    END IF;
    IF TG_OP = 'UPDATE' THEN
        SELECT "skillListingId" INTO new_parent_id FROM "SkillListingVersion" WHERE id = NEW."skillListingVersionId";
        IF catalogue_managed('skills', new_parent_id) OR EXISTS (
            SELECT 1 FROM "CatalogueReleaseVersion" WHERE "versionId" = NEW."skillListingVersionId") THEN
            RAISE EXCEPTION 'catalogue package files are immutable';
        END IF;
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END $$;
CREATE TRIGGER catalogue_file_guard BEFORE INSERT OR UPDATE OR DELETE ON "SkillListingFile"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_file();

CREATE FUNCTION guard_catalogue_history() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP <> 'INSERT' OR NOT catalogue_publisher_enabled() THEN
        RAISE EXCEPTION 'catalogue history is immutable and publisher-owned';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER catalogue_release_guard BEFORE INSERT OR UPDATE OR DELETE ON "CatalogueRelease"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_history();
CREATE TRIGGER catalogue_release_version_guard BEFORE INSERT OR UPDATE OR DELETE ON "CatalogueReleaseVersion"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_history();
CREATE TRIGGER catalogue_activation_guard BEFORE INSERT OR UPDATE OR DELETE ON "CatalogueActivation"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_history();

CREATE FUNCTION guard_catalogue_state() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP <> 'UPDATE' OR NOT catalogue_publisher_enabled() OR NEW.id <> OLD.id THEN
        RAISE EXCEPTION 'catalogue state is publisher-owned';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER catalogue_state_guard BEFORE INSERT OR UPDATE OR DELETE ON "CatalogueState"
FOR EACH ROW EXECUTE FUNCTION guard_catalogue_state();

DO $$
DECLARE guard RECORD; target_schema TEXT := current_schema();
BEGIN
    FOR guard IN SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS arguments
        FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = target_schema AND p.proname IN (
            'catalogue_managed', 'catalogue_publisher_enabled', 'guard_catalogue_parent',
            'guard_catalogue_assignment', 'guard_catalogue_version', 'guard_catalogue_file',
            'guard_catalogue_history', 'guard_catalogue_state')
    LOOP
        EXECUTE format('ALTER FUNCTION %I.%I(%s) SECURITY DEFINER', target_schema, guard.proname, guard.arguments);
        EXECUTE format('ALTER FUNCTION %I.%I(%s) SET search_path = pg_catalog, %I', target_schema, guard.proname, guard.arguments, target_schema);
        EXECUTE format('REVOKE ALL ON FUNCTION %I.%I(%s) FROM PUBLIC', target_schema, guard.proname, guard.arguments);
    END LOOP;
END $$;
