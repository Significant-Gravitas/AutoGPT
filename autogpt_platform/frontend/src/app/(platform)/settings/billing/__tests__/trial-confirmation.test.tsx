import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsConfirmTrialMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { TrialCheckoutConfirmation } from "@/components/organisms/TrialCard/TrialCheckoutConfirmation";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let searchParams = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useSearchParams: () => searchParams,
  usePathname: () => "/settings/billing",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

beforeEach(() => {
  setTrialUser();
  searchParams = new URLSearchParams("trial=success");
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({
        eligible: true,
        active: false,
        status: "checkout_pending",
      }),
    ),
  );
});
afterEach(() => setTrialUser(null));

function renderTrialReturn() {
  return render(
    <>
      <TrialCheckoutConfirmation />
      <TrialCard />
    </>,
  );
}

describe("trial Checkout return", () => {
  it("requires backend confirmation before displaying active trial access", async () => {
    const pending = deferredTrialResponse<ReturnType<typeof trialResponse>>();
    const confirm = vi.fn(() => pending.promise);
    server.use(getPostTrialsConfirmTrialMockHandler200(confirm));
    renderTrialReturn();
    expect((await screen.findByRole("status")).textContent).toMatch(
      /Confirming your trial/,
    );
    await screen.findByRole("button", { name: /start 7-day trial/i });
    expect(screen.queryByText("Your trial")).toBeNull();
    pending.resolve(trialResponse());
    expect(
      await screen.findByRole("button", { name: "Cancel trial" }),
    ).toBeDefined();
    expect(screen.queryByRole("status")).toBeNull();
    expect(confirm).toHaveBeenCalledOnce();
  });

  it("keeps a failed confirmation visible and lets the user retry", async () => {
    server.use(
      http.post("*/api/credits/trial/confirm", () =>
        HttpResponse.json(
          { detail: "Card setup is not complete yet" },
          { status: 409 },
        ),
      ),
    );
    renderTrialReturn();
    const retry = await screen.findByRole("button", { name: /try again/i });
    expect(screen.queryByText("Your trial")).toBeNull();
    server.use(getPostTrialsConfirmTrialMockHandler200(trialResponse()));
    fireEvent.click(retry);
    expect(
      await screen.findByRole("button", { name: "Cancel trial" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
  });

  it("does not treat an inactive 200 response as completed card setup", async () => {
    server.use(
      getPostTrialsConfirmTrialMockHandler200(
        trialResponse({ active: false, status: "trialing" }),
      ),
    );
    renderTrialReturn();
    expect(await screen.findByText(/Your trial is not active/)).toBeDefined();
    expect(screen.getByRole("button", { name: /try again/i })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Cancel trial" })).toBeNull();
  });

  it("accepts an already converted trial without showing another trial offer", async () => {
    const confirm = vi.fn(() =>
      trialResponse({ active: false, converted: true, status: "active" }),
    );
    server.use(getPostTrialsConfirmTrialMockHandler200(confirm));
    renderTrialReturn();
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
    await waitFor(() =>
      expect(
        screen.queryByRole("region", { name: "AutoGPT trial" }),
      ).toBeNull(),
    );
  });

  it("does not confirm on ordinary billing visits", async () => {
    searchParams = new URLSearchParams();
    const confirm = vi.fn(() => trialResponse());
    server.use(getPostTrialsConfirmTrialMockHandler200(confirm));
    renderTrialReturn();
    await screen.findByRole("button", { name: /start 7-day trial/i });
    expect(screen.queryByRole("status")).toBeNull();
    expect(confirm).not.toHaveBeenCalled();
  });

  it("cancels an older status request before publishing confirmed access", async () => {
    const pending = deferredTrialResponse<ReturnType<typeof trialResponse>>();
    let statusRequest: Request | undefined;
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(({ request }) => {
        statusRequest = request;
        return pending.promise;
      }),
      getPostTrialsConfirmTrialMockHandler200(async () => {
        await waitFor(() => expect(statusRequest).toBeDefined());
        return trialResponse();
      }),
    );
    try {
      renderTrialReturn();
      await screen.findByRole("button", { name: "Cancel trial" });
      expect(statusRequest?.signal.aborted).toBe(true);
    } finally {
      pending.resolve(
        trialResponse({
          eligible: true,
          active: false,
          status: "checkout_pending",
        }),
      );
    }
  });
});
