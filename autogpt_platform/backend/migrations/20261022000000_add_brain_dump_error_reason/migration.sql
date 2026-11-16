-- What decided a quality-gate rejection, beside errorCode
-- (quality.QualityRejection.reason). NULL for every other failure.
ALTER TABLE "OnboardingBrainDump"
    ADD COLUMN "errorReason" TEXT;
