import { http, HttpResponse, type JsonBodyType } from "msw";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";

import {
  getPatchV2UpdateFeedbackPromptMockHandler200,
  getPostV2ClaimFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import { setTrialUser } from "@/tests/integrations/trial-fixtures";

import SettingsBillingPage from "../page";

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

const mockSearchParams = { current: new URLSearchParams() };
const mockRouterReplace = vi.fn();
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useSearchParams: () => mockSearchParams.current,
  useRouter: () => ({
    push: vi.fn(),
    replace: mockRouterReplace,
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/settings/billing",
  useParams: () => ({}),
}));

const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

const TRIAL_TITLE = "Your free trial is ready when you are";
const UPGRADE_TITLE = "Your plan hasn't changed. What held you back?";

function jsonHandler(path: string, body: JsonBodyType) {
  return http.get(`*${path}`, () => HttpResponse.json(body));
}

function setup(params: Record<string, string>) {
  const claims: unknown[] = [];
  const patches: unknown[] = [];
  server.use(
    jsonHandler("/api/credits/subscription", {
      tier: "PRO",
      monthly_cost: 5000,
      has_active_stripe_subscription: true,
      status: "active",
    }),
    jsonHandler("/api/credits/manage", { url: "https://billing.stripe.com" }),
    jsonHandler("/api/credits", { credits: 1234 }),
    jsonHandler("/api/credits/transactions", {
      transactions: [],
      next_transaction_time: null,
    }),
    jsonHandler("/api/credits/auto-top-up", { amount: 0, threshold: 0 }),
    jsonHandler("/api/credits/invoices", []),
    jsonHandler("/api/v2/chat/copilot-usage", {
      daily: { percent_used: 0, resets_at: null },
      weekly: { percent_used: 0, resets_at: null },
    }),
    getPostV2ClaimFeedbackPromptMockHandler200(async ({ request }) => {
      claims.push(await request.clone().json());
      return { id: "prompt-1" };
    }),
    getPatchV2UpdateFeedbackPromptMockHandler200(async ({ request }) => {
      patches.push(await request.clone().json());
      return {
        id: "prompt-1",
        kind: "trial_checkout",
        answered: true,
        dismissed: false,
        reason: null,
        score: null,
      };
    }),
  );
  mockSearchParams.current = new URLSearchParams(params);
  return { claims, patches };
}

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  mockSearchParams.current = new URLSearchParams();
  mockRouterReplace.mockReset();
  toast.mockClear();
});

describe("returning from an abandoned checkout", () => {
  it("asks about a trial checkout and strips the param", async () => {
    const { claims } = setup({ trial: "cancelled" });
    render(<SettingsBillingPage />);
    expect(await screen.findByText(TRIAL_TITLE)).toBeDefined();
    expect(claims).toEqual([{ kind: "trial_checkout" }]);
    expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing");
  });

  it("asks about an upgrade checkout without a toast", async () => {
    const { claims } = setup({ subscription: "cancelled" });
    render(<SettingsBillingPage />);
    expect(await screen.findByText(UPGRADE_TITLE)).toBeDefined();
    expect(claims).toEqual([{ kind: "upgrade_checkout" }]);
    expect(toast).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Checkout cancelled" }),
    );
  });

  it("saves a picked reason", async () => {
    const { patches } = setup({ trial: "cancelled" });
    render(<SettingsBillingPage />);
    await screen.findByText(TRIAL_TITLE);
    fireEvent.click(
      screen.getByRole("button", { name: "My card didn't go through" }),
    );
    await waitFor(() => expect(patches).toEqual([{ reason: "card_failed" }]));
  });

  it("shows nothing on a successful return", async () => {
    const { claims } = setup({ subscription: "success" });
    render(<SettingsBillingPage />);
    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
    expect(claims).toEqual([]);
    expect(screen.queryByText(UPGRADE_TITLE)).toBeNull();
    expect(screen.queryByText(TRIAL_TITLE)).toBeNull();
  });

  it("records closing an unanswered checkout prompt", async () => {
    const { patches } = setup({ trial: "cancelled" });
    render(<SettingsBillingPage />);
    await screen.findByText(TRIAL_TITLE);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    await waitFor(() => expect(patches).toEqual([{ dismissed: true }]));
  });

  it("hides an open checkout prompt when the account changes", async () => {
    const { claims } = setup({ trial: "cancelled" });
    render(<SettingsBillingPage />);
    await screen.findByText(TRIAL_TITLE);
    setTrialUser("another-user");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(claims).toHaveLength(1);
  });

  it("claims once across re-renders", async () => {
    const { claims } = setup({ trial: "cancelled" });
    const { rerender } = render(<SettingsBillingPage />);
    await screen.findByText(TRIAL_TITLE);
    rerender(<SettingsBillingPage />);
    await screen.findByText(TRIAL_TITLE);
    expect(claims).toHaveLength(1);
  });
});
