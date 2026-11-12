-- The user's latest onboarding voice call, recorded on the user's onboarding
-- row when the call starts on the voice relay. NULL until the first call; a
-- later call replaces it. Call ids are unique, so the index is too.
ALTER TABLE "UserOnboarding"
    ADD COLUMN "voiceCallId" TEXT;

CREATE UNIQUE INDEX "UserOnboarding_voiceCallId_key"
    ON "UserOnboarding"("voiceCallId");
