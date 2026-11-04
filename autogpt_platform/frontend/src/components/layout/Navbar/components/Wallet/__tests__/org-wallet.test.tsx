import type { UserOnboarding } from "@/lib/autogpt-server-api";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Wallet } from "../Wallet";

const MANAGED_MESSAGE =
  "Your organization's credits are managed by its owner and billing managers.";
const TOP_UP_REFUSED =
  "A team organization's credits are added by AutoGPT for now, so they can't be bought here.";

const TEAM_ORG = {
  id: "org-team",
  name: "Acme Inc",
  slug: "acme",
  avatarUrl: null,
  isPersonal: false,
  memberCount: 3,
};

const PERSONAL_ORG = {
  ...TEAM_ORG,
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  isPersonal: true,
  memberCount: 1,
};

const onboardingState = vi.hoisted(() => ({
  completedSteps: ["ONBOARDING_COMPLETE"],
  walletShown: true,
  notified: [],
  rewardedFor: [],
  usageReason: null,
  integrations: [],
  otherIntegrations: null,
  selectedStoreListingVersionId: null,
  agentInput: null,
  onboardingAgentExecutionId: null,
  lastRunAt: null,
  consecutiveRunDays: 0,
  agentRuns: 0,
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => children,
  useOnboarding: () => ({
    state: onboardingState as unknown as UserOnboarding,
    updateState: vi.fn(),
    step: 1,
    setStep: vi.fn(),
    completeStep: vi.fn(),
  }),
}));

// The real context hands out one client, and the websocket effect is keyed on
// it; the wallet's credits go through useCredits' own client, not this one.
const backendAPI = vi.hoisted(() => ({
  onWebSocketMessage: () => () => {},
  connectWebSocket: () => {},
}));

vi.mock("@/lib/autogpt-server-api/context", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/autogpt-server-api/context")>();
  return { ...actual, useBackendAPI: () => backendAPI };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: vi.fn(() => true) };
});

vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({ isPaymentEnabled: true, isResolved: true }),
}));

function seedActiveOrg(org: typeof TEAM_ORG) {
  useOrgTeamStore.setState({
    activeOrgID: org.id,
    activeTeamID: null,
    orgs: [org],
    teams: [],
    isLoaded: true,
  });
}

function answerCredits(answer: { credits: number } | "forbidden") {
  const orgHeaders: (string | null)[] = [];
  server.use(
    http.get("*/api/credits", ({ request }) => {
      orgHeaders.push(request.headers.get("X-Org-Id"));
      return answer === "forbidden"
        ? HttpResponse.json(
            { detail: "Missing org permission: MANAGE_BILLING" },
            { status: 403 },
          )
        : HttpResponse.json(answer);
    }),
  );
  return orgHeaders;
}

async function openWallet() {
  await userEvent.click(
    await screen.findByRole("button", { name: "Usage and credits" }),
  );
  return screen.findByText("Automation credits");
}

describe("Wallet in a personal organization", () => {
  beforeEach(() => seedActiveOrg(PERSONAL_ORG));

  it("shows the balance and Add credits, as before", async () => {
    const orgHeaders = answerCredits({ credits: 979 });
    render(<Wallet compact />);

    await openWallet();
    expect(await screen.findByText("$9.79")).toBeDefined();
    expect(screen.getByRole("button", { name: "Add credits" })).toBeDefined();
    expect(screen.queryByText(TOP_UP_REFUSED)).toBeNull();
    expect(screen.queryByText(MANAGED_MESSAGE)).toBeNull();
    // The request still carries the active (personal) organization the way
    // every request does.
    expect(new Set(orgHeaders)).toEqual(new Set([PERSONAL_ORG.id]));
  });
});

describe("Wallet in a team organization", () => {
  beforeEach(() => seedActiveOrg(TEAM_ORG));

  it("shows a billing manager the organization's balance, with no way to buy credits", async () => {
    const orgHeaders = answerCredits({ credits: 4200 });
    render(<Wallet compact />);

    await openWallet();
    expect(await screen.findByText("$42.00")).toBeDefined();
    expect(screen.getByText(TOP_UP_REFUSED)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Add credits" })).toBeNull();
    expect(orgHeaders.length).toBeGreaterThan(0);
    expect(new Set(orgHeaders)).toEqual(new Set([TEAM_ORG.id]));
  });

  it("tells a plain member who manages the credits instead of hiding the wallet", async () => {
    const orgHeaders = answerCredits("forbidden");
    render(<Wallet compact />);

    await waitFor(() => {
      expect(orgHeaders).toContain(TEAM_ORG.id);
    });
    await openWallet();
    expect(await screen.findByText(MANAGED_MESSAGE)).toBeDefined();
    expect(screen.queryByText("-")).toBeNull();
    expect(screen.queryByText(TOP_UP_REFUSED)).toBeNull();
    expect(screen.queryByRole("button", { name: "Add credits" })).toBeNull();
  });
});
