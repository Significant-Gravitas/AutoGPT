-- This migration converts the stats column from a list to an object.
UPDATE "AgentGraphExecution"
SET "stats" = (stats::jsonb -> 0)::text
WHERE stats IS NOT NULL AND jsonb_typeof(stats::jsonb) = 'array';
