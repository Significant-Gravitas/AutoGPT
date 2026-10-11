import { http, HttpResponse } from "msw";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getGetV2GetCopilotUsageMockHandler200,
  getGetV2GetCopilotUsageResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { server } from "@/mocks/mock-server";
import { act, render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { WalletCopilotUsage } from "../components/WalletCopilotUsage";

const future = new Date("2030-09-17T15:00:00Z");

function mockUsage(
  daily: number | null = 32,
  weekly: number | null = 18,
  tier: CoPilotUsagePublic["tier"] = "PRO",
) {
  server.use(
    getGetV2GetCopilotUsageMockHandler200(
      getGetV2GetCopilotUsageResponseMock200({
        daily:
          daily === null ? null : { percent_used: daily, resets_at: future },
        weekly:
          weekly === null ? null : { percent_used: weekly, resets_at: future },
        tier,
        reset_cost: 0,
      }),
    ),
  );
}

afterEach(() => {
  vi.useRealTimers();
  setTrialUser(null);
});

describe("WalletCopilotUsage", () => {
  it("shows the plan and the independent daily and weekly allowances", async () => {
    mockUsage();
    render(<WalletCopilotUsage />);

    expect(screen.getByText("Chat usage")).toBeDefined();
    expect(
      screen.getByText(
        "Your allowance for conversations with Otto and experts.",
      ),
    ).toBeDefined();
    expect(await screen.findByText("Pro plan")).toBeDefined();
    expect(
      screen
        .getByRole("progressbar", { name: "Today usage" })
        .getAttribute("aria-valuenow"),
    ).toBe("32");
    expect(
      screen
        .getByRole("progressbar", { name: "This week usage" })
        .getAttribute("aria-valuenow"),
    ).toBe("18");
  });

  it("keeps loading distinct from zero usage", () => {
    mockUsage();
    render(<WalletCopilotUsage />);

    expect(
      screen.getByRole("status", { name: "Loading chat usage" }),
    ).toBeDefined();
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(screen.queryByText("0% used")).toBeNull();
  });

  it("shows unlimited only after a successful response with no limits", async () => {
    mockUsage(null, null);
    render(<WalletCopilotUsage />);

    expect(await screen.findByText("No usage limits")).toBeDefined();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("shows a configured weekly limit without inventing a daily allowance", async () => {
    mockUsage(null, 81);
    render(<WalletCopilotUsage />);

    expect(await screen.findByText("81% used")).toBeDefined();
    expect(screen.queryByText("Today")).toBeNull();
  });

  it("lets the user recover from a usage request failure", async () => {
    server.use(
      http.get(
        "*/api/chat/usage",
        () => new HttpResponse(null, { status: 503 }),
      ),
    );
    render(<WalletCopilotUsage />);

    expect(
      await screen.findByText("Usage is unavailable right now."),
    ).toBeDefined();
    expect(screen.queryByRole("progressbar")).toBeNull();
    mockUsage(42, 25);
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("42% used")).toBeDefined();
  });

  it("shows the trial's total allowance and end date instead of recurring resets", async () => {
    setTrialUser();
    mockUsage(12, 18, "TRIAL");
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({ allowance_used_percent: 42.4 }),
      ),
    );
    render(<WalletCopilotUsage />);

    const bar = await screen.findByRole("progressbar", {
      name: "Trial allowance usage",
    });
    expect(bar.getAttribute("aria-valuenow")).toBe("42.4");
    expect(screen.getByText("42% used")).toBeDefined();
    expect(screen.getByText(/^Ends /)).toBeDefined();
    expect(screen.queryByText("Today")).toBeNull();
    expect(screen.queryByText("This week")).toBeNull();
    expect(screen.queryByText(/^Resets /)).toBeNull();
  });

  it("does not round a nearly exhausted trial to fully used", async () => {
    setTrialUser();
    mockUsage(99.9, 99.9, "TRIAL");
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({ allowance_used_percent: 99.9 }),
      ),
    );
    render(<WalletCopilotUsage />);

    expect(await screen.findByText("99.9% used")).toBeDefined();
    expect(screen.queryByText("100% used")).toBeNull();
  });

  it("refreshes trial allowance while the wallet stays open", async () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    setTrialUser();
    mockUsage(0, 0, "TRIAL");
    let allowanceUsed = 12;
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(() =>
        trialResponse({ allowance_used_percent: allowanceUsed }),
      ),
    );
    render(<WalletCopilotUsage />);
    expect(await screen.findByText("12% used")).toBeDefined();

    allowanceUsed = 54;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    expect(await screen.findByText("54% used")).toBeDefined();
    expect(screen.queryByText("12% used")).toBeNull();
  });

  it("keeps unavailable trial usage distinct from its daily and weekly windows", async () => {
    setTrialUser();
    mockUsage(0, 0, "TRIAL");
    server.use(
      http.get(
        "*/api/credits/trial",
        () => new HttpResponse(null, { status: 503 }),
      ),
    );
    render(<WalletCopilotUsage />);

    expect(
      await screen.findByText("Usage is unavailable right now."),
    ).toBeDefined();
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(screen.queryByText("0% used")).toBeNull();
  });
});
