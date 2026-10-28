from typing import LiteralString

# A listing version is served on the public marketplace while it is approved
# and neither it nor its listing is deleted: superseded approved versions are
# re-activated when a newer version is rejected, and get_store_agent_details
# serves an active version even when it is not available. Expects aliases
# ``slv`` (StoreListingVersion) and ``sl`` (StoreListing); never NULL.
VERSION_IS_PUBLIC: LiteralString = """(
    slv."submissionStatus" = 'APPROVED'
    AND NOT slv."isDeleted"
    AND NOT sl."isDeleted"
)"""

# A creator profile is served publicly while its owner has a non-deleted
# listing that has (or had) an approved version, or a live skill listing.
# Expects alias ``p`` (Profile); EXISTS keeps it non-NULL when a listing has
# no active version.
CREATOR_IS_PUBLIC: LiteralString = """(
    EXISTS (
        SELECT 1
        FROM platform."StoreListing" AS public_listing
        WHERE public_listing."owningUserId" = p."userId"
          AND NOT public_listing."isDeleted"
          AND (
              public_listing."hasApprovedVersion"
              OR EXISTS (
                  SELECT 1
                  FROM platform."StoreListingVersion" AS public_version
                  WHERE public_version."storeListingId" = public_listing.id
                    AND public_version."submissionStatus" = 'APPROVED'
                    AND NOT public_version."isDeleted"
              )
          )
    )
    OR EXISTS (
        SELECT 1
        FROM platform."SkillListing" AS skill
        JOIN platform."SkillListingVersion" AS skill_version
          ON skill_version.id = skill."activeVersionId"
        WHERE skill."owningUserId" = p."userId"
          AND NOT skill."isDeleted"
          AND skill."hasApprovedVersion"
          AND skill_version."submissionStatus" = 'APPROVED'
          AND skill_version."isAvailable"
          AND NOT skill_version."isDeleted"
    )
)"""

# Org avatars have no single owner. The private media endpoint serves a file to
# members of an active org its uploader belongs to, so these rows are rewritten
# only when the uploader is an active member of that org, and the UPDATE takes
# no owner parameter.
PATH_OWNER_TARGETS = frozenset(
    {
        "Organization.avatarUrl",
        "OrganizationProfile.avatarUrl",
    }
)

UPDATE_QUERIES: dict[str, LiteralString] = {
    "Profile.avatarUrl": f"""
        UPDATE platform."Profile" AS p
        SET "avatarUrl" = $2
        WHERE p.id = $1
          AND p."avatarUrl" = $3
          AND p."userId" = $4
          AND NOT {CREATOR_IS_PUBLIC}
    """,
    "Expert.avatarUrl": """
        UPDATE platform."Expert" AS e
        SET "avatarUrl" = $2
        WHERE e.id = $1
          AND e."avatarUrl" = $3
          AND e."ownerUserId" = $4
    """,
    "LibraryAgent.imageUrl": """
        UPDATE platform."LibraryAgent" AS la
        SET "imageUrl" = $2
        WHERE la.id = $1
          AND la."imageUrl" = $3
          AND la."userId" = $4
    """,
    "StoreListingVersion.imageUrls": f"""
        UPDATE platform."StoreListingVersion" AS slv
        SET "imageUrls" = $2::text[]
        FROM platform."StoreListing" AS sl
        WHERE slv.id = $1
          AND slv."imageUrls" = $3::text[]
          AND sl.id = slv."storeListingId"
          AND sl."owningUserId" = $4
          AND NOT {VERSION_IS_PUBLIC}
    """,
    "StoreListingVersion.videoUrl": f"""
        UPDATE platform."StoreListingVersion" AS slv
        SET "videoUrl" = $2
        FROM platform."StoreListing" AS sl
        WHERE slv.id = $1
          AND slv."videoUrl" = $3
          AND sl.id = slv."storeListingId"
          AND sl."owningUserId" = $4
          AND NOT {VERSION_IS_PUBLIC}
    """,
    "StoreListingVersion.agentOutputDemoUrl": f"""
        UPDATE platform."StoreListingVersion" AS slv
        SET "agentOutputDemoUrl" = $2
        FROM platform."StoreListing" AS sl
        WHERE slv.id = $1
          AND slv."agentOutputDemoUrl" = $3
          AND sl.id = slv."storeListingId"
          AND sl."owningUserId" = $4
          AND NOT {VERSION_IS_PUBLIC}
    """,
    "Organization.avatarUrl": """
        UPDATE platform."Organization" AS o
        SET "avatarUrl" = $2
        WHERE o.id = $1
          AND o."avatarUrl" = $3
    """,
    "OrganizationProfile.avatarUrl": """
        UPDATE platform."OrganizationProfile" AS op
        SET "avatarUrl" = $2
        WHERE op."organizationId" = $1
          AND op."avatarUrl" = $3
    """,
}

# Publishing only swaps a reference for the public copy of the same object,
# so a compare-and-swap on the old value is the whole guard.
PUBLISH_UPDATE_QUERIES: dict[str, LiteralString] = {
    "StoreListingVersion.imageUrls": """
        UPDATE platform."StoreListingVersion"
        SET "imageUrls" = $2::text[]
        WHERE id = $1 AND "imageUrls" = $3::text[]
    """,
    "StoreListingVersion.videoUrl": """
        UPDATE platform."StoreListingVersion"
        SET "videoUrl" = $2
        WHERE id = $1 AND "videoUrl" = $3
    """,
    "StoreListingVersion.agentOutputDemoUrl": """
        UPDATE platform."StoreListingVersion"
        SET "agentOutputDemoUrl" = $2
        WHERE id = $1 AND "agentOutputDemoUrl" = $3
    """,
    "Profile.avatarUrl": """
        UPDATE platform."Profile"
        SET "avatarUrl" = $2
        WHERE id = $1 AND "avatarUrl" = $3
    """,
    "LibraryAgent.imageUrl": """
        UPDATE platform."LibraryAgent"
        SET "imageUrl" = $2
        WHERE id = $1 AND "imageUrl" = $3
    """,
    "OAuthApplication.logoUrl": """
        UPDATE platform."OAuthApplication"
        SET "logoUrl" = $2
        WHERE id = $1 AND "logoUrl" = $3
    """,
}
