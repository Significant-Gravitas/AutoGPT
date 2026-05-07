-- Add isHidden flag to LibraryAgent for trigger agents.
-- Trigger agents are hidden from the main library listing; their parent
-- agents are derived from AgentExecutorBlock usage in the trigger graph.
ALTER TABLE "LibraryAgent" ADD COLUMN "isHidden" BOOLEAN NOT NULL DEFAULT false;

-- Index for filtering by hidden status in library listings.
-- All library list queries filter on (userId, isDeleted, isArchived, isHidden).
CREATE INDEX "LibraryAgent_isHidden_idx" ON "LibraryAgent" ("userId", "isHidden") WHERE "isDeleted" = false AND "isArchived" = false;
