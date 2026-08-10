-- Onboarding voice "brain dump": the raw recording plus its full
-- transcript.  Postgres enums (rather than TEXT) so an invalid status or
-- input mode is rejected at the DB layer; new states need a cheap
-- ``ALTER TYPE ... ADD VALUE``.
CREATE TYPE "BrainDumpStatus" AS ENUM (
    'recording_uploaded',
    'transcribing',
    'transcribed',
    'extracting',
    'completed',
    'failed'
);

CREATE TYPE "BrainDumpInputMode" AS ENUM ('voice', 'typed', 'skipped');

-- One row per user: re-recording overwrites the take in place, so
-- ``userId`` is unique and ``recordingId`` identifies the current take
-- (it is what makes ``finalize`` idempotent across retries).
--
-- ``transcript`` holds the COMPLETE transcript even when a shortened
-- version is what gets injected into the copilot's first prompt —
-- truncation is never silent.  ``audioPath`` is kept on failure so the
-- download endpoint keeps working and the pipeline can be re-run.
CREATE TABLE "OnboardingBrainDump" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "recordingId" TEXT NOT NULL,
    "status" "BrainDumpStatus" NOT NULL,
    "inputMode" "BrainDumpInputMode" NOT NULL,
    "audioPath" TEXT,
    "mimeType" TEXT,
    "sizeBytes" INTEGER,
    "durationSecs" DOUBLE PRECISION,
    "transcript" TEXT,
    "transcriptLang" TEXT,
    "errorCode" TEXT,

    CONSTRAINT "OnboardingBrainDump_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "OnboardingBrainDump_userId_key" ON "OnboardingBrainDump"("userId");

CREATE INDEX "OnboardingBrainDump_userId_idx" ON "OnboardingBrainDump"("userId");

ALTER TABLE "OnboardingBrainDump"
    ADD CONSTRAINT "OnboardingBrainDump_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
