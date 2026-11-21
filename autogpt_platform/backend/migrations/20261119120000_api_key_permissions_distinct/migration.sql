-- An API key takes each permission at most once. Keys stored before that
-- bound keep one copy of each permission they hold.
UPDATE "APIKey"
SET "permissions" = ARRAY(SELECT DISTINCT p FROM unnest("permissions") AS p ORDER BY p)
WHERE cardinality("permissions") > (
    SELECT count(DISTINCT p) FROM unnest("APIKey"."permissions") AS p
);
