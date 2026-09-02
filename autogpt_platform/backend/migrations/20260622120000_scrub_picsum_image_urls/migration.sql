-- Scrub legacy picsum.photos placeholder URLs persisted in the database.
--
-- These were written by the old random picsum fallback (removed in this change)
-- and by old seed data. Now that picsum.photos is dropped from the Next.js image
-- domain allow-list, next/image throws on these unconfigured-host URLs at render
-- time (uncatchable by onError), crashing the card via the error boundary.
--
-- Real, user-chosen images (any non-picsum URL) are left untouched. Only the
-- random picsum placeholders are removed.

-- StoreListingVersion.imageUrls (String[]): drop any picsum entries, keeping the
-- order of the remaining real images. Only touch rows that actually contain one.
UPDATE "StoreListingVersion"
SET "imageUrls" = ARRAY(
    SELECT url
    FROM unnest("imageUrls") WITH ORDINALITY AS u(url, ord)
    WHERE url NOT LIKE '%picsum.photos%'
    ORDER BY ord
)
WHERE EXISTS (
    SELECT 1 FROM unnest("imageUrls") AS url WHERE url LIKE '%picsum.photos%'
);

-- Profile.avatarUrl: empty string (NOT NULL) — the Creator view types avatar_url
-- as a non-nullable String, so a NULL here would break GET /api/store/creators.
-- Empty renders the frontend's fallback avatar.
UPDATE "Profile"
SET "avatarUrl" = ''
WHERE "avatarUrl" LIKE '%picsum.photos%';

-- LibraryAgent.imageUrl: nullable with no dependent non-null view → NULL it out.
UPDATE "LibraryAgent"
SET "imageUrl" = NULL
WHERE "imageUrl" LIKE '%picsum.photos%';
