-- The chat session of the user's onboarding interview, recorded on the user's
-- onboarding row the first time the interview is opened, so every later open
-- finds the same session. NULL until then. One interview per session id, so
-- the index is unique. No foreign key, so the database keeps the id when
-- the session row goes; the application clears it when it deletes the
-- interview session. A start that finds no session at the id creates the
-- interview there again, and one that finds a chat that is not the
-- interview records a fresh id.
ALTER TABLE "UserOnboarding"
    ADD COLUMN "interviewSessionId" TEXT;

CREATE UNIQUE INDEX "UserOnboarding_interviewSessionId_key"
    ON "UserOnboarding"("interviewSessionId");
