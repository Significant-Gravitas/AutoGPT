-- The onboarding voice call whose words became the user's interview's take,
-- written when the pipeline accepts the words of a call that ran to its close
-- or that the user confirmed. NULL while no call has taken the interview.
-- Call ids are unique, so the index is too.
ALTER TABLE "UserOnboarding"
    ADD COLUMN "voiceInterviewCallId" TEXT;

CREATE UNIQUE INDEX "UserOnboarding_voiceInterviewCallId_key"
    ON "UserOnboarding"("voiceInterviewCallId");
