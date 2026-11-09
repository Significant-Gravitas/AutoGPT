-- The chat session of the user's onboarding interview, recorded on the user's
-- onboarding row the first time the interview is opened, so every later open
-- finds the same session. NULL until then. One interview per session id, so
-- the index is unique. No foreign key: a deleted session leaves the id, and
-- the next start creates the interview again at it.
ALTER TABLE "UserOnboarding"
    ADD COLUMN "interviewSessionId" TEXT;

CREATE UNIQUE INDEX "UserOnboarding_interviewSessionId_key"
    ON "UserOnboarding"("interviewSessionId");
