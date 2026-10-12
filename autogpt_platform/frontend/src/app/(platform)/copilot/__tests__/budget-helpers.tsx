import {
  getGetV2GetChatShareStateMockHandler200,
  getGetV2GetCopilotUsageMockHandler200,
  getGetV2GetPendingMessagesMockHandler200,
  getGetV2ListChatConnectionsMockHandler200,
  getGetV2ListChatTransportsMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetSubscriptionStatusMockHandler200 } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { AtResetTurn } from "@/app/api/__generated__/models/atResetTurn";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import type { SubscriptionTier } from "@/app/api/__generated__/models/subscriptionTier";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { ProActivationProvider } from "@/services/pro-activation/ProActivationProvider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { CopilotChatHost } from "../CopilotChatHost";
import { sessionHandler, TEST_SESSION_ID } from "./sse-helpers";

// Local wall-clock times, so the copy reads the same in any time zone.
export const SATURDAY_1AM = new Date(2026, 9, 10, 1, 0);
export const SATURDAY_848PM = new Date(2026, 9, 10, 20, 48);
export const SUNDAY_1AM = new Date(2026, 9, 11, 1, 0);
export const MONDAY_1AM = new Date(2026, 9, 12, 1, 0);
export const MONDAY_107AM = new Date(2026, 9, 12, 1, 7);

export function weeklyExhausted(
  over: Partial<CoPilotUsagePublic> = {},
): CoPilotUsagePublic {
  return {
    tier: "MAX",
    daily: { percent_used: 100, resets_at: SUNDAY_1AM },
    weekly: { percent_used: 100, resets_at: MONDAY_1AM },
    reset_cost: 0,
    state: "exhausted",
    limiting_window: "weekly",
    resets_at: MONDAY_1AM,
    ...over,
  };
}

const NOT_A_TRIAL: TrialStatusResponse = {
  active: false,
  converted: true,
  eligible: false,
  offer: null,
  ends_at: null,
  status: "active",
};

export function connection(
  provider: "platform" | "codex" | "microsoft_365_copilot",
): AIConnectionOffer {
  const names = {
    platform: "AutoGPT Platform",
    codex: "ChatGPT",
    microsoft_365_copilot: "Microsoft 365 Copilot",
  };
  return {
    offer_id:
      provider === "platform" ? "platform:deployment" : `${provider}:c1`,
    provider_family: provider,
    auth_provider: provider,
    display_name: names[provider],
    auth_method: provider === "platform" ? "deployment" : "oauth",
    credential_id: provider === "platform" ? null : "c1",
    backed_by_label: "",
    description: "",
    state: "ready",
    selectable: true,
    is_default: provider === "platform",
    tiers: [],
    limitations: [],
    lock_reason: null,
    unlock_href: null,
  };
}

interface ChatSetup {
  usage: CoPilotUsagePublic;
  provider?: "platform" | "codex" | "microsoft_365_copilot";
  offers?: AIConnectionOffer[];
  trial?: TrialStatusResponse;
}

/** The endpoints a chat reads around its AutoPilot budget, pinned. */
export function budgetHandlers({
  usage,
  provider = "platform",
  offers = [connection("platform")],
  trial = NOT_A_TRIAL,
}: ChatSetup) {
  return [
    sessionHandler({
      metadata: {
        dry_run: false,
        builder_graph_id: null,
        llm_auth_provider: provider,
      },
    }),
    getGetV2GetCopilotUsageMockHandler200(usage),
    getGetSubscriptionStatusMockHandler200({
      tier: (usage.tier ?? "PRO") as SubscriptionTier,
      monthly_cost: 0,
      tier_costs: { PRO: 5000, MAX: 32000 },
      proration_credit_cents: 0,
    }),
    getGetTrialsGetTrialStatusMockHandler200(trial),
    getGetV2GetPendingMessagesMockHandler200({ count: 0, messages: [] }),
    getGetV2ListChatConnectionsMockHandler200({ offers }),
    getGetV2GetChatShareStateMockHandler200({
      is_shared: false,
      share_token: null,
      auto_share_executions: false,
      message_count: 0,
      linked_run_count: 0,
      file_count: 0,
    }),
    getGetV2ListChatTransportsMockHandler200({
      transports: [
        {
          auth_provider: "platform",
          credential_id: null,
          label: "AutoGPT",
          available: true,
          default: true,
        },
      ],
    }),
  ];
}

/** The at-reset endpoints, backed by a list the test can read. */
export function fakeAtReset(jobs: AtResetTurn[] = []) {
  const state = {
    jobs: [...jobs],
    scheduled: [] as { sessionId: string; body: unknown }[],
    cancelled: [] as string[],
    refusal: null as "not_out_of_budget" | "no_reset" | null,
    // The longest a single message may be; the real server allows 64,000.
    maxMessageChars: Infinity,
  };
  const handlers = [
    http.get("*/api/chat/sessions/:sessionId/at-reset", () =>
      HttpResponse.json({ jobs: state.jobs }),
    ),
    http.post(
      "*/api/chat/sessions/:sessionId/at-reset",
      async ({ request, params }) => {
        const body = (await request.json()) as {
          kind: "send";
          message: string;
        };
        state.scheduled.push({ sessionId: String(params.sessionId), body });
        if (state.refusal) {
          return HttpResponse.json({ detail: state.refusal }, { status: 409 });
        }
        // As the server does: a send joins the message already waiting.
        const waiting = state.jobs.find((j) => j.kind === body.kind);
        const message = waiting?.message
          ? `${waiting.message}\n\n${body.message}`
          : body.message;
        if (message.length > state.maxMessageChars) {
          return HttpResponse.json(
            { detail: "message_too_long" },
            { status: 413 },
          );
        }
        const job = { kind: body.kind, run_at: MONDAY_107AM, message };
        state.jobs = [...state.jobs.filter((j) => j.kind !== body.kind), job];
        return HttpResponse.json(job);
      },
    ),
    http.delete(
      "*/api/chat/sessions/:sessionId/at-reset/:kind",
      ({ params }) => {
        state.cancelled.push(String(params.kind));
        state.jobs = state.jobs.filter((job) => job.kind !== params.kind);
        return new HttpResponse(null, { status: 204 });
      },
    ),
  ];
  return { state, handlers };
}

export function renderChat(
  searchParams: string = `?sessionId=${TEST_SESSION_ID}`,
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Providers({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <NuqsTestingAdapter searchParams={searchParams} hasMemory>
          <BackendAPIProvider>
            <OnboardingProvider>
              <TooltipProvider>
                <ProActivationProvider>{children}</ProActivationProvider>
              </TooltipProvider>
            </OnboardingProvider>
          </BackendAPIProvider>
        </NuqsTestingAdapter>
      </QueryClientProvider>
    );
  }
  const view = render(
    <CopilotChatHost droppedFiles={[]} onDroppedFilesConsumed={() => {}} />,
    { wrapper: Providers },
  );
  return { ...view, queryClient };
}
