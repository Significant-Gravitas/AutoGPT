import { getAdminMock } from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { getAnalyticsMock } from "@/app/api/__generated__/endpoints/analytics/analytics.msw";
import { getApiKeysMock } from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";
import { getAuthMock } from "@/app/api/__generated__/endpoints/auth/auth.msw";
import { getBlocksMock } from "@/app/api/__generated__/endpoints/blocks/blocks.msw";
import {
  getCompleteBrainDumpGreetingMockHandler200,
  getGetBrainDumpIntroMockHandler200,
  getGetBrainDumpRecommendedProvidersMockHandler200,
  getGetBrainDumpSuggestedTasksMockHandler200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { getBriefingsMock } from "@/app/api/__generated__/endpoints/briefings/briefings.msw";
import {
  getChatMock,
  getGetV2ListTurnsAtBudgetResetMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getCreditsMock } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { getDefaultMock } from "@/app/api/__generated__/endpoints/default/default.msw";
import { getEmailMock } from "@/app/api/__generated__/endpoints/email/email.msw";
import { getGetFeatureFlagsMockHandler } from "@/app/api/__generated__/endpoints/feature-flags/feature-flags.msw";
import {
  getExecutionsMock,
  getGetV2GetAnsweredReviewsWaitingToRunForChatSessionMockHandler,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { getGetExperimentsGetExperimentAssignmentMockHandler } from "@/app/api/__generated__/endpoints/experiments/experiments.msw";
import {
  getExpertsMock,
  getGetExpertOnboardingMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
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
import { getSearchMock } from "@/app/api/__generated__/endpoints/search/search.msw";
import { getSkillsMock } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { getStoreMock } from "@/app/api/__generated__/endpoints/store/store.msw";
import { getWorkspaceMock } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { http, HttpResponse } from "msw";

// Pass hard-coded data to individual handler functions to override faker-generated data.
export const mockHandlers = [
  // No activation in progress unless a test explicitly provides one.
  http.get("/api/proxy/api/credits/pro-activation/current", () =>
    HttpResponse.json({ detail: "No activation in progress" }, { status: 404 }),
  ),
  // No published releases unless a test provides some.
  http.get("/api/changelog", () => HttpResponse.text("")),
  http.get("/api/changelog/latest", () => HttpResponse.json({ releases: [] })),
  // No flag vendor unless a test provides one, so every flag serves its default.
  getGetFeatureFlagsMockHandler({ flags: {}, configured: false }),
  ...getAdminMock(),
  ...getAnalyticsMock(),
  ...getApiKeysMock(),
  ...getAuthMock(),
  ...getBlocksMock(),
  // Greeting already done by default so the onboarding greeting stays out
  // of unrelated tests; greeting tests override with their own handler.
  getGetBrainDumpIntroMockHandler200({
    path: "A",
    greeting: "",
    prompts: [],
    greeting_done: true,
  }),
  getCompleteBrainDumpGreetingMockHandler200({ greeting_done: true }),
  // No suggested first tasks unless a test sets some up.
  getGetBrainDumpSuggestedTasksMockHandler200({ tasks: [], stored: false }),
  // Ready by default so the preparing step's recommendation gate never
  // holds unrelated tests; gate tests override with ready: false.
  getGetBrainDumpRecommendedProvidersMockHandler200({
    ready: true,
    providers: [],
  }),
  ...getBriefingsMock(),
  // Nothing set to send at a budget reset unless a test sets it up; random
  // jobs would put a scheduled-message chip above every chat's composer.
  getGetV2ListTurnsAtBudgetResetMockHandler200({ jobs: [] }),
  ...getChatMock(),
  ...getCreditsMock(),
  ...getDefaultMock(),
  ...getEmailMock(),
  // No answers waiting to run unless a test sets them up; random rows would
  // seed receipts into every chat's approval box.
  getGetV2GetAnsweredReviewsWaitingToRunForChatSessionMockHandler([]),
  ...getExecutionsMock(),
  // Nobody is enrolled by default, so every test sees the control experience.
  getGetExperimentsGetExperimentAssignmentMockHandler(null),
  // No setup questions waiting unless a test sets some up; a random chat id
  // would pull every empty expert chat into that chat.
  getGetExpertOnboardingMockHandler200(null),
  ...getExpertsMock(),
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
  ...getSearchMock(),
  ...getSkillsMock(),
  ...getStoreMock(),
  ...getWorkspaceMock(),
];
