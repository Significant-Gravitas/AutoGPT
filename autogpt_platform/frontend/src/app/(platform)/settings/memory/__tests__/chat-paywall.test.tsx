import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetMyMemoryOverviewMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { PAYWALL_TOAST_KEY } from "@/components/contextual/PaywallToast/PaywallToast";
import { server } from "@/mocks/mock-server";

const toastSpy = vi.hoisted(() => vi.fn());
const captureException = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

vi.mock("@sentry/nextjs", () => ({
  captureMessage: vi.fn(),
  captureException,
  addBreadcrumb: vi.fn(),
  getTraceData: () => ({}),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

import SettingsMemoryPage from "../page";

function answerSessionCreate(status: number, detail: string) {
  server.use(
    http.post("*/api/chat/sessions", () =>
      HttpResponse.json({ detail }, { status }),
    ),
  );
}

async function openSummaryChat() {
  render(<SettingsMemoryPage />);
  await userEvent
    .setup()
    .click(await screen.findByRole("button", { name: "View my summary" }));
}

beforeEach(() => {
  resetCopilotChatRegistry();
  toastSpy.mockClear();
  captureException.mockClear();
  window.history.replaceState({}, "", "/settings/memory");
  server.use(
    getListExpertsMockHandler200([]),
    getListMyMemoryFactsMockHandler200({ expert_id: null, items: [] }),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 0,
      entities: 0,
      episodes: 0,
    }),
  );
});

describe("Memory chat start", () => {
  it("offers plans to an account without one and reports no error", async () => {
    answerSessionCreate(
      402,
      "A subscription is required to use this feature. Upgrade to continue.",
    );

    await openSummaryChat();

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "The memory chat needs an active plan",
          replaceKey: PAYWALL_TOAST_KEY,
        }),
      );
    });
    await waitFor(() => {
      expect(
        screen.queryByRole("complementary", { name: "Memory chat panel" }),
      ).toBeNull();
    });
    expect(screen.queryByText("Could not start the memory chat")).toBeNull();
    expect(captureException).not.toHaveBeenCalled();
  });

  it("shows the start error and reports a real failure", async () => {
    answerSessionCreate(500, "Internal server error");

    await openSummaryChat();

    expect(
      await screen.findByText("Could not start the memory chat"),
    ).toBeDefined();
    expect(captureException).toHaveBeenCalledOnce();
    expect(captureException.mock.calls[0][0]).toMatchObject({ status: 500 });
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ replaceKey: PAYWALL_TOAST_KEY }),
    );
  });
});
