import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { SubscriptionTab } from "../components/SubscriptionTab/SubscriptionTab";

vi.mock("../components/SubscriptionTab/YourPlanCard/YourPlanCard", () => ({
  YourPlanCard: () => <div data-testid="paid-plan">Paid plan</div>,
}));

beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

it("does not mount the paid plan while initial trial status is pending", async () => {
  const pending = deferredTrialResponse<ReturnType<typeof trialResponse>>();
  server.use(getGetTrialsGetTrialStatusMockHandler200(() => pending.promise));
  render(<SubscriptionTab />);
  expect(screen.queryByTestId("paid-plan")).toBeNull();
  pending.resolve(trialResponse());
  await screen.findByRole("button", { name: "Cancel trial" });
  expect(screen.queryByTestId("paid-plan")).toBeNull();
});

it("retains the paid plan fallback when trial status fails", async () => {
  server.use(
    http.get("*/api/credits/trial", () =>
      HttpResponse.json({ detail: "Unavailable" }, { status: 503 }),
    ),
  );
  render(<SubscriptionTab />);
  await screen.findByRole("button", { name: /try again/i });
  expect(screen.getByTestId("paid-plan")).toBeDefined();
});
