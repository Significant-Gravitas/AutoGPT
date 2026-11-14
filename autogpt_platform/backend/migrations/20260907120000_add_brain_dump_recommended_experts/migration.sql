-- The team AutoPilot proposes on the greeting page: a diagnosis, the
-- roster templates worth hiring first, and an optional "raise your own"
-- suggestion for a need no template covers. Written by a background job
-- beside the greeting one, so a slow or failed team never delays the
-- greeting. NULL means "not generated yet"; a stored `source` of
-- "disabled" records a run the feature flag switched off.
ALTER TABLE "OnboardingBrainDump"
    ADD COLUMN "recommendedExperts" JSONB;
