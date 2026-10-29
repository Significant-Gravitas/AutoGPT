-- Provider recommendations for the welcome dialog's "Connect your tools"
-- panel. Written by a background job that runs beside (not inside) the
-- greeting pipeline, so a slow or failed recommendation never delays the
-- greeting. NULL means "not generated yet" — an empty list is a real
-- result ("nothing worth recommending").
ALTER TABLE "OnboardingBrainDump"
    ADD COLUMN "recommendedProviders" JSONB;
