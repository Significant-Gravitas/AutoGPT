import {
  getGetV2GetCopilotUsageMockHandler200,
  getGetV2GetCopilotUsageResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetSubscriptionStatusMockHandler200,
  getGetSubscriptionStatusResponseMock200,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { setTrialUser } from "@/tests/integrations/trial-fixtures";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UsageCreditsDialog } from "../UsageCreditsDialog";

const flags = vi.hoisted(() => ({ payment: true }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: vi.fn((flag: string) =>
      flag === actual.Flag.ENABLE_PLATFORM_PAYMENT ? flags.payment : false,
    ),
  };
});

const window = {
  percent_used: 10,
  resets_at: new Date("2030-01-01T00:00:00Z"),
};

function mockPlan(tier: "PRO" | "MAX", limits: Partial<CoPilotUsagePublic>) {
  server.use(
    getGetV2GetCopilotUsageMockHandler200(
      getGetV2GetCopilotUsageResponseMock200({
        tier,
        daily: null,
        weekly: null,
        reset_cost: 0,
        ...limits,
      }),
    ),
    getGetSubscriptionStatusMockHandler200(
      getGetSubscriptionStatusResponseMock200({ tier }),
    ),
  );
}

beforeEach(() => {
  flags.payment = true;
  setTrialUser("paid-user");
});

afterEach(() => setTrialUser(null));

// SECRT-3133 review: the dialog said the same thing to everyone, so it pointed
// at an Add credits button some wallets hide, and told plans without limits
// that they reset daily and weekly.
describe("UsageCreditsDialog", () => {
  it("tells a plan without limits what the billing page tells it", async () => {
    mockPlan("MAX", {});
    render(<UsageCreditsDialog isOpen onOpenChange={() => {}} />);

    expect(
      await screen.findByText(/which has no daily or weekly limits\./),
    ).toBeDefined();
    expect(screen.queryByText(/resets daily/)).toBeNull();
  });

  it("names the windows a capped plan resets on", async () => {
    mockPlan("PRO", { daily: window, weekly: window });
    render(<UsageCreditsDialog isOpen onOpenChange={() => {}} />);

    expect(await screen.findByText(/resets daily and weekly\./)).toBeDefined();
  });

  it("leaves out buying and the card when the wallet can't buy credits", async () => {
    flags.payment = false;
    mockPlan("PRO", { daily: window, weekly: window });
    render(<UsageCreditsDialog isOpen onOpenChange={() => {}} />);

    expect(
      await screen.findByText("What are automation credits for?"),
    ).toBeDefined();
    expect(screen.queryByText("When is my card charged?")).toBeNull();
    expect(screen.queryByText(/Add credits/)).toBeNull();
    expect(
      screen.getByText(/finish the setup steps in the wallet\./),
    ).toBeDefined();
  });

  it("answers the card question first with the reassurance", async () => {
    mockPlan("PRO", { daily: window, weekly: window });
    render(<UsageCreditsDialog isOpen onOpenChange={() => {}} />);

    expect(
      await screen.findByText(
        /^A chat or a step never charges your card directly\. Your card is charged when your plan starts and each time it renews/,
      ),
    ).toBeDefined();
  });
});
