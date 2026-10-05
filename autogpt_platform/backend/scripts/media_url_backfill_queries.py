from typing import LiteralString

UPDATE_QUERIES: dict[str, LiteralString] = {
    "Profile.avatarUrl": """
        UPDATE platform."Profile" AS p
        SET "avatarUrl" = $2, "updatedAt" = NOW()
        WHERE p.id = $1
          AND p."avatarUrl" = $3
          AND p."userId" = $4
          AND NOT EXISTS (
              SELECT 1
              FROM platform."StoreListing" AS sl
              JOIN platform."StoreListingVersion" AS active
                ON active.id = sl."activeVersionId"
              WHERE sl."owningUserId" = p."userId"
                AND active."submissionStatus" = 'APPROVED'
                AND NOT sl."isDeleted"
                AND sl."hasApprovedVersion"
                AND NOT active."isDeleted"
                AND active."isAvailable"
          )
    """,
    "Expert.avatarUrl": """
        UPDATE platform."Expert" AS e
        SET "avatarUrl" = $2, "updatedAt" = NOW()
        WHERE e.id = $1
          AND e."avatarUrl" = $3
          AND e."ownerUserId" = $4
    """,
    "LibraryAgent.imageUrl": """
        UPDATE platform."LibraryAgent" AS la
        SET "imageUrl" = $2, "updatedAt" = NOW()
        WHERE la.id = $1
          AND la."imageUrl" = $3
          AND la."userId" = $4
          AND la."isCreatedByUser"
          AND la."creatorId" IS NULL
    """,
    "StoreListingVersion.imageUrls": """
        UPDATE platform."StoreListingVersion" AS slv
        SET "imageUrls" = $2::text[], "updatedAt" = NOW()
        FROM platform."StoreListing" AS sl
        WHERE slv.id = $1
          AND slv."imageUrls" = $3::text[]
          AND sl.id = slv."storeListingId"
          AND sl."owningUserId" = $4
          AND NOT (
              sl."activeVersionId" = slv.id
              AND slv."submissionStatus" = 'APPROVED'
              AND NOT sl."isDeleted"
              AND sl."hasApprovedVersion"
              AND NOT slv."isDeleted"
              AND slv."isAvailable"
          )
    """,
    "StoreListingVersion.videoUrl": """
        UPDATE platform."StoreListingVersion" AS slv
        SET "videoUrl" = $2, "updatedAt" = NOW()
        FROM platform."StoreListing" AS sl
        WHERE slv.id = $1
          AND slv."videoUrl" = $3
          AND sl.id = slv."storeListingId"
          AND sl."owningUserId" = $4
          AND NOT (
              sl."activeVersionId" = slv.id
              AND slv."submissionStatus" = 'APPROVED'
              AND NOT sl."isDeleted"
              AND sl."hasApprovedVersion"
              AND NOT slv."isDeleted"
              AND slv."isAvailable"
          )
    """,
    "StoreListingVersion.agentOutputDemoUrl": """
        UPDATE platform."StoreListingVersion" AS slv
        SET "agentOutputDemoUrl" = $2, "updatedAt" = NOW()
        FROM platform."StoreListing" AS sl
        WHERE slv.id = $1
          AND slv."agentOutputDemoUrl" = $3
          AND sl.id = slv."storeListingId"
          AND sl."owningUserId" = $4
          AND NOT (
              sl."activeVersionId" = slv.id
              AND slv."submissionStatus" = 'APPROVED'
              AND NOT sl."isDeleted"
              AND sl."hasApprovedVersion"
              AND NOT slv."isDeleted"
              AND slv."isAvailable"
          )
    """,
}
