-- The retire CLI also rewrites presets' overrides of an LLM-block node and
-- records those rows here, so --revert can restore them.
ALTER TABLE "LlmModelMigration" ADD COLUMN "migratedPresetInputIds" JSONB NOT NULL DEFAULT '[]';
