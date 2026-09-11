-- Delete every persona phrase the roster seed ever wrote as a skill (none has a
-- SKILL.md behind it); an expert owning a skill folder of that slug keeps it.
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
           FROM   UNNEST(e."skills") WITH ORDINALITY AS t("name", "position")
           WHERE  t."name" <> ALL (p."list")
              OR  EXISTS (
                    SELECT 1
                    FROM   "UserWorkspaceFile" f
                    JOIN   "UserWorkspace" w ON w."id" = f."workspaceId"
                    WHERE  w."userId" = e."ownerUserId"
                      AND  NOT f."isDeleted"
                      AND  f."path" = '/experts/' || e."id" || '/skills/'
                                      || LOWER(t."name") || '/SKILL.md'
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
