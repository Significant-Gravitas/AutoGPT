ALTER TABLE "SkillVersion" ADD COLUMN "packageManifest" JSONB;

CREATE TABLE "SkillFileContent" (
    "userId" TEXT NOT NULL,
    "sha256" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "content" BYTEA NOT NULL,

    CONSTRAINT "SkillFileContent_pkey" PRIMARY KEY ("userId", "sha256")
);

ALTER TABLE "SkillFileContent" ADD CONSTRAINT "SkillFileContent_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
