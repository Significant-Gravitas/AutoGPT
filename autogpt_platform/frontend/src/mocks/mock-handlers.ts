import { getAdminMock } from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { getAnalyticsMock } from "@/app/api/__generated__/endpoints/analytics/analytics.msw";
import { getApiKeysMock } from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";
import { getAuthMock } from "@/app/api/__generated__/endpoints/auth/auth.msw";
import { getBlocksMock } from "@/app/api/__generated__/endpoints/blocks/blocks.msw";
import { getChatMock } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getCreditsMock } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { getDefaultMock } from "@/app/api/__generated__/endpoints/default/default.msw";
import { getEmailMock } from "@/app/api/__generated__/endpoints/email/email.msw";
import { getExecutionsMock } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { getFilesMock } from "@/app/api/__generated__/endpoints/files/files.msw";
import { getGraphsMock } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getHealthMock } from "@/app/api/__generated__/endpoints/health/health.msw";
import { getIntegrationsMock } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { getLibraryMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getMetricsMock } from "@/app/api/__generated__/endpoints/metrics/metrics.msw";
import { getOauthMock } from "@/app/api/__generated__/endpoints/oauth/oauth.msw";
import { getOnboardingMock } from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import { getOttoMock } from "@/app/api/__generated__/endpoints/otto/otto.msw";
import { getPresetsMock } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getSchedulesMock } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getStoreMock } from "@/app/api/__generated__/endpoints/store/store.msw";

// Pass hard-coded data to individual handler functions to override faker-generated data.
export const mockHandlers = [
  ...getAdminMock(),
  ...getAnalyticsMock(),
  ...getApiKeysMock(),
  ...getAuthMock(),
  ...getBlocksMock(),
  ...getChatMock(),
  ...getCreditsMock(),
  ...getDefaultMock(),
  ...getEmailMock(),
  ...getExecutionsMock(),
  ...getFilesMock(),
  ...getGraphsMock(),
  ...getHealthMock(),
  ...getIntegrationsMock(),
  ...getLibraryMock(),
  ...getMetricsMock(),
  ...getOauthMock(),
  ...getOnboardingMock(),
  ...getOttoMock(),
  ...getPresetsMock(),
  ...getSchedulesMock(),
  ...getStoreMock(),
];
