-- Email sign-up stored the address's local part as the name ("jane.doe87"), as did
-- the Supabase copy for accounts without one. Sign-up now stores an empty name when
-- none is given; this clears the old auto-fills. An identity with any OAuth account
-- keeps its name, which may have come from that provider's profile.
UPDATE "UserAuthIdentity" i
SET "name" = ''
WHERE i."name" <> ''
  AND lower(i."name") = lower(split_part(i."email", '@', 1))
  AND NOT EXISTS (
    SELECT 1 FROM "UserAuthAccount" a
    WHERE a."userId" = i."id" AND a."providerId" <> 'credential'
  );

UPDATE "User" u
SET "name" = NULL
WHERE u."name" IS NOT NULL
  AND lower(u."name") = lower(split_part(u."email", '@', 1))
  AND NOT EXISTS (
    SELECT 1 FROM "UserAuthAccount" a
    WHERE a."userId" = u."id" AND a."providerId" <> 'credential'
  );
