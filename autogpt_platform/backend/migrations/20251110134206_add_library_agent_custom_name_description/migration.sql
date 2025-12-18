-- AlterTable
ALTER TABLE "LibraryAgent" ADD COLUMN "name" TEXT,
ADD COLUMN "description" TEXT;

-- AddComment
COMMENT ON COLUMN "LibraryAgent"."name" IS 'Optional custom name override (e.g., from marketplace). If set, takes precedence over AgentGraph name.';
COMMENT ON COLUMN "LibraryAgent"."description" IS 'Optional custom description override (e.g., from marketplace). If set, takes precedence over AgentGraph description.';
