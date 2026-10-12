-- Delete every persona phrase the roster seed ever wrote as a skill (none has a
-- SKILL.md behind it); a phrase whose SKILL.md does exist is a real assignment.
WITH "phrases" AS (
  SELECT ARRAY[
    'Content strategy', 'Social copy', 'SEO writing', 'Web copy', 'Positioning',
    'Email campaigns', 'Prospecting', 'Lead qualification', 'Contact research',
    'Cold outreach', 'ICP targeting', 'Account research', 'Meeting prep',
    'Follow-ups', 'Support triage', 'Scheduling', 'Checklists'
  ]::TEXT[] AS "list"
),
"cleaned" AS (
  SELECT e."id",
         ARRAY(
           SELECT t."name"
           FROM   UNNEST(e."skills") WITH ORDINALITY AS t("name", "position"),
                  -- The row may carry a display name ("Content strategy") for the
                  -- folder "content-strategy": skill_name_key()'s key, in SQL.
                  LATERAL (
                    SELECT regexp_replace(BTRIM(LOWER(t."name")), '[\s_-]+', '-', 'g')
                  ) AS k("slug")
           WHERE  t."name" <> ALL (p."list")
              OR  EXISTS (
                    SELECT 1
                    FROM   "UserWorkspaceFile" f
                    JOIN   "UserWorkspace" w ON w."id" = f."workspaceId"
                    WHERE  w."userId" = e."ownerUserId"
                      AND  NOT f."isDeleted"
                      -- The expert's own copy, or the owner-library file that the
                      -- per-expert ownership backfill copies on the next cold
                      -- listing — deleting the name is what would strand it.
                      AND  f."path" IN (
                             '/experts/' || e."id" || '/skills/' || k."slug" || '/SKILL.md',
                             '/skills/' || k."slug" || '/SKILL.md'
                           )
                  )
           ORDER BY t."position"
         ) AS "skills"
  FROM   "Expert" e, "phrases" p
  WHERE  e."skills" && p."list"
)
UPDATE "Expert" e
SET    "skills" = c."skills"
FROM   "cleaned" c
WHERE  e."id" = c."id"
  AND  e."skills" IS DISTINCT FROM c."skills";
