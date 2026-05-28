-- Creator-tooling search: keep ILIKE '%q%' fast as creator libraries grow.
-- Plain B-tree indexes can't help LIKE patterns with a leading wildcard, so
-- we install pg_trgm and add GIN indexes on the columns that the new
-- /submissions and /my-unpublished-agents search queries scan (Prisma
-- `contains` + `mode: "insensitive"` compiles to ILIKE). StoreSubmission is
-- a view over StoreListing and StoreListingVersion, so the /submissions
-- indexes live on those base table columns instead of the view itself.
--
-- CREATE INDEX uses CONCURRENTLY so prod writes are not blocked during the
-- build. CONCURRENTLY cannot run inside a transaction block; Prisma runs
-- each statement in a migration file on its own connection without wrapping
-- the file in BEGIN/COMMIT, so each statement below is committed
-- independently and can use CONCURRENTLY.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX CONCURRENTLY IF NOT EXISTS "AgentGraph_name_trgm_idx"
    ON "AgentGraph" USING gin (name gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS "AgentGraph_description_trgm_idx"
    ON "AgentGraph" USING gin (description gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS "StoreListingVersion_name_trgm_idx"
    ON "StoreListingVersion" USING gin (name gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS "StoreListingVersion_subHeading_trgm_idx"
    ON "StoreListingVersion" USING gin ("subHeading" gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS "StoreListing_slug_trgm_idx"
    ON "StoreListing" USING gin (slug gin_trgm_ops);
