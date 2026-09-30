import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

beforeEach(() => setTrialUser("user-a"));
afterEach(() => setTrialUser(null));

it("requires confirmation before irreversibly ending a trial", async () => {
  const cancel = vi.fn(() =>
    trialResponse({ active: false, status: "canceled" }),
  );
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
    getPostTrialsCancelTrialMockHandler200(cancel),
  );
  render(<TrialCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Cancel trial" }));
  expect(
    await screen.findByRole("dialog", { name: "End your trial now?" }),
  ).toBeDefined();
  expect(cancel).not.toHaveBeenCalled();
  expect(screen.getByText(/You cannot restart this trial/)).toBeDefined();
  fireEvent.click(screen.getByRole("button", { name: "Keep trial" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(cancel).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel trial" }));
  fireEvent.click(await screen.findByRole("button", { name: "End trial now" }));
  await screen.findByText(/Cancellation confirmed/);
  expect(cancel).toHaveBeenCalledOnce();
});

it("dismisses confirmation on Escape without canceling", async () => {
  const cancel = vi.fn(() => trialResponse());
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
    getPostTrialsCancelTrialMockHandler200(cancel),
  );
  render(<TrialCard />);
  fireEvent.click(await screen.findByRole("button", { name: "Cancel trial" }));
  const dialog = await screen.findByRole("dialog");
  fireEvent.keyDown(dialog, { key: "Escape" });
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(cancel).not.toHaveBeenCalled();
});

it("does not carry an open cancellation confirmation across accounts", async () => {
  const cancel = vi.fn(() => trialResponse());
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
    getPostTrialsCancelTrialMockHandler200(cancel),
  );
  render(<TrialCard />);
  await screen.findByRole("button", { name: "Cancel trial" });
  act(() => setTrialUser("user-b"));
  await screen.findByRole("button", { name: "Cancel trial" });
  fireEvent.click(screen.getByRole("button", { name: "Cancel trial" }));
  await screen.findByRole("dialog");
  act(() => setTrialUser("user-a"));
  await screen.findByRole("button", { name: "Cancel trial" });
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(cancel).not.toHaveBeenCalled();
});

it.each([
  ["BASIC", "Basic"],
  ["PRO", "Pro"],
  ["MAX", "Max"],
  ["BUSINESS", "Team"],
] as const)(
  "uses the %s plan's display name and masks personalized billing copy",
  async (tier, label) => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({
          eligible: true,
          active: false,
          offer: { ...trialOffer, tier },
        }),
      ),
    );
    render(<TrialCard />);
    await screen.findByText(`Try AutoGPT ${label} for 7 days`);
    expect(
      screen
        .getByText(/Card required. No subscription charge today/)
        .classList.contains("sentry-unmask"),
    ).toBe(false);
  },
);
