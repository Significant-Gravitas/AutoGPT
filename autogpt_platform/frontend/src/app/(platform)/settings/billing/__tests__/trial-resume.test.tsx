import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
  getPostTrialsResumeTrialMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const posthog = vi.hoisted(() => ({ capture: vi.fn() }));
vi.mock("@posthog/react", () => ({
  usePostHog: () => posthog,
  useFeatureFlagVariantKey: () => undefined,
}));

const DAY = 24 * 60 * 60 * 1000;
const endsAt = new Date("2030-09-17T15:00:00Z");
const active = trialResponse({ cancel_keeps_access: true });
const cancelPending = trialResponse({
  cancel_keeps_access: true,
  cancel_at_period_end: true,
});

beforeEach(() => {
  setTrialUser();
  posthog.capture.mockClear();
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date(endsAt.getTime() - 5 * DAY));
});
afterEach(() => {
  vi.useRealTimers();
  setTrialUser(null);
});

async function openCanceledPopup() {
  fireEvent.click(await screen.findByRole("button", { name: "Cancel trial" }));
  const confirm = await screen.findByRole("dialog", {
    name: "Cancel your trial?",
  });
  fireEvent.click(
    within(confirm).getByRole("button", { name: "Cancel trial" }),
  );
  return screen.findByRole("dialog", { name: "Cancellation confirmed" });
}

async function expectNormalActiveTrial() {
  expect(
    await screen.findByRole("button", { name: "Cancel trial" }),
  ).toBeDefined();
  expect(screen.getByText(/Your trial ends.*\$20\.00/)).toBeDefined();
  expect(screen.queryByText("Cancellation pending")).toBeNull();
  expect(screen.queryByRole("button", { name: "Resume trial" })).toBeNull();
}

describe("resuming a cancel-pending trial", () => {
  it("resumes from the popup, closes it and restores the active trial", async () => {
    const pending = deferredTrialResponse<typeof active>();
    const resume = vi.fn(() => pending.promise);
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(active),
      getPostTrialsCancelTrialMockHandler200(cancelPending),
      getPostTrialsResumeTrialMockHandler200(resume),
    );
    render(<TrialCard />);
    const popup = await openCanceledPopup();
    const button = within(popup).getByRole("button", { name: "Resume trial" });
    fireEvent.click(button);
    await waitFor(() => expect(resume).toHaveBeenCalledOnce());
    expect(button.hasAttribute("disabled")).toBe(true);
    fireEvent.click(button);
    pending.resolve(active);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await expectNormalActiveTrial();
    expect(resume).toHaveBeenCalledOnce();
    expect(
      posthog.capture.mock.calls.filter(
        ([event]) => event === "trial_resume_clicked",
      ),
    ).toEqual([["trial_resume_clicked", { days_left: 5 }]]);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("resumes from the cancel-pending card", async () => {
    const resume = vi.fn(() => active);
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(cancelPending),
      getPostTrialsResumeTrialMockHandler200(resume),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Resume trial" }),
    );
    await expectNormalActiveTrial();
    expect(resume).toHaveBeenCalledOnce();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("reports a failed resume, closes the popup and refetches the status", async () => {
    const status = vi.fn(() => active);
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(status),
      getPostTrialsCancelTrialMockHandler200(cancelPending),
      http.post("*/api/credits/trial/resume", () =>
        HttpResponse.json({ detail: "Stripe is unavailable" }, { status: 502 }),
      ),
    );
    render(<TrialCard />);
    const popup = await openCanceledPopup();
    expect(status).toHaveBeenCalledOnce();
    status.mockImplementation(() => cancelPending);
    fireEvent.click(
      within(popup).getByRole("button", { name: "Resume trial" }),
    );
    expect((await screen.findByRole("alert")).textContent).toContain(
      "Stripe is unavailable",
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(status).toHaveBeenCalledTimes(2);
    expect(
      screen
        .getByRole("button", { name: "Resume trial" })
        .hasAttribute("disabled"),
    ).toBe(false);
    expect(screen.getByText("Cancellation pending")).toBeDefined();
  });
});
