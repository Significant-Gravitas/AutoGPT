-- CreateTable
CREATE TABLE "Agents" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "name" TEXT,
    "description" TEXT,
    "author" TEXT,
    "keywords" TEXT[],
    "categories" TEXT[],
    "search" tsvector DEFAULT ''::tsvector,
    "graph" JSONB NOT NULL,

    CONSTRAINT "Agents_pkey" PRIMARY KEY ("id","version")
);

-- CreateTable
CREATE TABLE "AnalyticsTracker" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "agentId" UUID NOT NULL,
    "views" INTEGER NOT NULL,
    "downloads" INTEGER NOT NULL
);

-- CreateIndex
CREATE UNIQUE INDEX "Agents_id_key" ON "Agents"("id");

-- CreateIndex
CREATE UNIQUE INDEX "AnalyticsTracker_id_key" ON "AnalyticsTracker"("id");

-- AddForeignKey
ALTER TABLE "AnalyticsTracker" ADD CONSTRAINT "AnalyticsTracker_agentId_fkey" FOREIGN KEY ("agentId") REFERENCES "Agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;


-- Add trigger to update the search column with the tsvector of the agent
-- Function to be invoked by trigger

CREATE OR REPLACE FUNCTION update_tsvector_column() RETURNS TRIGGER AS $$

BEGIN

NEW.search := to_tsvector('english', COALESCE(NEW.description, '')|| ' ' ||COALESCE(NEW.name, '')|| ' ' ||COALESCE(NEW.author, ''));

RETURN NEW;

END;

$$ LANGUAGE plpgsql SECURITY definer SET search_path = public, pg_temp;

-- Trigger that keeps the TSVECTOR up to date

DROP TRIGGER IF EXISTS "update_tsvector" ON "Agents";

CREATE TRIGGER "update_tsvector"

BEFORE INSERT OR UPDATE ON "Agents"

FOR EACH ROW

EXECUTE FUNCTION update_tsvector_column ();