import { http, HttpResponse, type JsonBodyType } from "msw";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  onTestFinished,
  vi,
} from "vitest";

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
        kind: "upgrade_checkout",
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

const UPGRADE_RETURN = {
  subscription: "cancelled",
  plan: "PRO",
  cycle: "monthly",
};

async function sendReason(name: string) {
  await screen.findByText(UPGRADE_TITLE);
  fireEvent.click(screen.getByRole("button", { name }));
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
}

describe("returning from an abandoned checkout", () => {
  it("asks nothing after leaving a trial checkout", async () => {
    const { claims } = setup({ trial: "cancelled" });
    render(<SettingsBillingPage />);
    await screen.findByText("Billing & usage");
    expect(claims).toEqual([]);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("asks about an upgrade checkout without a toast", async () => {
    const { claims } = setup(UPGRADE_RETURN);
    render(<SettingsBillingPage />);
    expect(await screen.findByText(UPGRADE_TITLE)).toBeDefined();
    expect(claims).toEqual([{ kind: "upgrade_checkout" }]);
    expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing");
    expect(toast).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Checkout cancelled" }),
    );
  });

  it("offers just looking as an answer", async () => {
    const { patches } = setup(UPGRADE_RETURN);
    render(<SettingsBillingPage />);
    await screen.findByText(UPGRADE_TITLE);
    fireEvent.click(
      screen.getByRole("button", { name: "Just looking for now" }),
    );
    await waitFor(() => expect(patches).toEqual([{ reason: "just_looking" }]));
  });

  it("opens the same checkout again after a failed card", async () => {
    const originalLocation = window.location;
    const location = {
      origin: "http://localhost",
      pathname: "/settings/billing",
      href: "http://localhost/settings/billing",
    };
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: location,
    });
    onTestFinished(() => {
      Object.defineProperty(window, "location", {
        configurable: true,
        writable: true,
        value: originalLocation,
      });
    });
    const tierRequests: unknown[] = [];
    setup(UPGRADE_RETURN);
    server.use(
      http.post("*/api/credits/subscription", async ({ request }) => {
        tierRequests.push(await request.clone().json());
        return HttpResponse.json({ url: "https://checkout.stripe.com/retry" });
      }),
    );
    render(<SettingsBillingPage />);
    await sendReason("My card didn't go through");
    fireEvent.click(
      await screen.findByRole("button", { name: "Try another card" }),
    );
    await waitFor(() => expect(tierRequests).toHaveLength(1));
    expect(tierRequests[0]).toMatchObject({
      tier: "PRO",
      billing_cycle: "monthly",
      surface: "billing",
    });
    await waitFor(() =>
      expect(location.href).toBe("https://checkout.stripe.com/retry"),
    );
  });

  it("offers another card only to a card answer", async () => {
    setup(UPGRADE_RETURN);
    render(<SettingsBillingPage />);
    await sendReason("The price");
    expect(await screen.findByText(/we read every one/)).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Try another card" }),
    ).toBeNull();
  });

  it("shows nothing on a successful return", async () => {
    const { claims } = setup({ subscription: "success" });
    render(<SettingsBillingPage />);
    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
    expect(claims).toEqual([]);
    expect(screen.queryByText(UPGRADE_TITLE)).toBeNull();
  });

  it("records closing an unanswered checkout prompt", async () => {
    const { patches } = setup(UPGRADE_RETURN);
    render(<SettingsBillingPage />);
    await screen.findByText(UPGRADE_TITLE);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    await waitFor(() => expect(patches).toEqual([{ dismissed: true }]));
  });

  it("hides an open checkout prompt when the account changes", async () => {
    const { claims } = setup(UPGRADE_RETURN);
    render(<SettingsBillingPage />);
    await screen.findByText(UPGRADE_TITLE);
    setTrialUser("another-user");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(claims).toHaveLength(1);
  });

  it("claims once across re-renders", async () => {
    const { claims } = setup(UPGRADE_RETURN);
    const { rerender } = render(<SettingsBillingPage />);
    await screen.findByText(UPGRADE_TITLE);
    rerender(<SettingsBillingPage />);
    await screen.findByText(UPGRADE_TITLE);
    expect(claims).toHaveLength(1);
  });
});
