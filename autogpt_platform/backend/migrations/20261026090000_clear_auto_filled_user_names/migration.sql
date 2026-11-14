-- Email sign-up stored the address's local part as the name ("jane.doe87"), as did
-- the Supabase copy for accounts without one. Sign-up now stores an empty name when
-- none is given; this clears the old auto-fills. A name stays where an OAuth account
-- may have given it, which one linked to an email sign-up later cannot: linking leaves
-- the name as it was.
UPDATE "UserAuthIdentity" i
SET "name" = ''
WHERE i."name" <> ''
  AND lower(i."name") = lower(split_part(i."email", '@', 1))
  AND NOT EXISTS (
    SELECT 1 FROM "UserAuthAccount" a
    WHERE a."userId" = i."id" AND a."providerId" <> 'credential'
      -- An hour's margin, since the Supabase copy dates a password account to the
      -- user's creation, a moment before the account of a Google sign-up.
      AND NOT EXISTS (
        SELECT 1 FROM "UserAuthAccount" c
        WHERE c."userId" = i."id" AND c."providerId" = 'credential'
          AND c."createdAt" < a."createdAt" - interval '1 hour'
      )
  );

UPDATE "User" u
SET "name" = NULL
WHERE u."name" IS NOT NULL
  AND lower(u."name") = lower(split_part(u."email", '@', 1))
  AND NOT EXISTS (
    SELECT 1 FROM "UserAuthAccount" a
    WHERE a."userId" = u."id" AND a."providerId" <> 'credential'
      AND NOT EXISTS (
        SELECT 1 FROM "UserAuthAccount" c
        WHERE c."userId" = u."id" AND c."providerId" = 'credential'
          AND c."createdAt" < a."createdAt" - interval '1 hour'
      )
  );

-- A name given to Otto or a Tally form was kept only in AutoPilot's understanding, whose copy
-- Otto no longer reads; it fills an empty account name unless it is the auto-filled local part.
UPDATE "User" u
SET "name" = btrim(c."data" ->> 'name')
FROM "CoPilotUnderstanding" c
WHERE c."userId" = u."id"
  AND (u."name" IS NULL OR u."name" = '')
  AND btrim(c."data" ->> 'name') <> ''
  AND lower(btrim(c."data" ->> 'name')) <> lower(split_part(u."email", '@', 1));
