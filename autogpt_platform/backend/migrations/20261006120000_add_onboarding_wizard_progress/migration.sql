ALTER TABLE "UserOnboarding"
    ADD COLUMN "wizardProgress" JSONB,
    ADD COLUMN "wizardRevision" INTEGER NOT NULL DEFAULT 0;
