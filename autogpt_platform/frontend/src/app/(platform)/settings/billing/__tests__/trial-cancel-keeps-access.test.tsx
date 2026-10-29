import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { COUNTRIES } from "@/components/molecules/PlanCard/countries";
import { PLANS } from "@/components/molecules/PlanCard/plans";
import { SubscriptionPlans } from "@/components/organisms/SubscriptionPlans/SubscriptionPlans";
import {
  formatTrialEnd,
  formatTrialEndDate,
} from "@/components/organisms/TrialCard/helpers";
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
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const endsAt = new Date("2030-09-17T15:00:00Z");

beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

describe("cancel confirmation", () => {
  it("promises access until the trial ends and no charge", async () => {
    const cancel = vi.fn(() => trialResponse({ cancel_at_period_end: true }));
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
      getPostTrialsCancelTrialMockHandler200(cancel),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Cancel trial" }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Cancel your trial?",
    });
    expect(dialog.textContent).toContain(
      "Your trial won't convert to a paid plan and your card won't be charged.",
    );
    expect(dialog.textContent).toContain(
      `You keep full access until ${formatTrialEnd(endsAt)}, and you can resume your trial any time before then.`,
    );
    expect(within(dialog).getByText(formatTrialEnd(endsAt)).tagName).toBe(
      "STRONG",
    );
    expect(within(dialog).queryByText(/immediately|cannot restart/)).toBeNull();
    fireEvent.click(within(dialog).getByRole("button", { name: "Keep trial" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(cancel).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Cancel trial" }));
    const confirm = await screen.findByRole("dialog", {
      name: "Cancel your trial?",
    });
    fireEvent.click(
      within(confirm).getByRole("button", { name: "Cancel trial" }),
    );
    await screen.findByText("Cancellation pending");
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("shows the ended trial when the cancel finds it already over", async () => {
    let ended = false;
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(() =>
        ended
          ? trialResponse({ active: false, status: "canceled" })
          : trialResponse(),
      ),
      http.post("*/api/credits/trial/cancel", () => {
        ended = true;
        return HttpResponse.json(
          { detail: "This trial has ended. Manage the plan in billing." },
          { status: 409 },
        );
      }),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Cancel trial" }),
    );
    const confirm = await screen.findByRole("dialog", {
      name: "Cancel your trial?",
    });
    fireEvent.click(
      within(confirm).getByRole("button", { name: "Cancel trial" }),
    );
    expect((await screen.findByRole("alert")).textContent).toContain(
      "This trial has ended. Manage the plan in billing.",
    );
    expect(await screen.findByText("Your trial has ended")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Cancel trial" })).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("drops the ends-immediately warning from the active trial", async () => {
    server.use(getGetTrialsGetTrialStatusMockHandler200(trialResponse()));
    render(<TrialCard />);
    await screen.findByRole("button", { name: "Cancel trial" });
    expect(screen.getByText(/Your trial ends.*\$20\.00/)).toBeDefined();
    expect(screen.queryByText(/immediately/)).toBeNull();
  });
});

describe("cancel-pending trial card", () => {
  it("shows the pending cancellation and a resume action", async () => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({ cancel_at_period_end: true }),
      ),
    );
    render(<TrialCard />);
    expect(await screen.findByText("Cancellation pending")).toBeDefined();
    expect(screen.getByRole("heading", { name: "Your trial" })).toBeDefined();
    const card = screen.getByRole("region", { name: "AutoGPT trial" });
    expect(card.textContent).toContain(
      `Cancellation confirmed. Your trial will not convert to a paid plan and your card won't be charged. Trial access ends ${formatTrialEnd(endsAt)}.`,
    );
    expect(card.textContent).toContain(
      `Resume to keep the trial and start Pro on ${formatTrialEndDate(endsAt)} at $20 / month, plus applicable tax.`,
    );
    expect(screen.getByRole("button", { name: "Resume trial" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Cancel trial" })).toBeNull();
    expect(screen.queryByText(/immediately/)).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("trial terms before the trial starts", () => {
  it("states the cancel terms on the offer", async () => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({ eligible: true, active: false, status: null }),
      ),
    );
    render(<TrialCard />);
    await screen.findByRole("button", { name: /start 7-day trial/i });
    expect(
      screen.getByText(
        "Trial usage is limited. If you cancel, you keep access until the trial ends. You can manage your plan in billing.",
      ),
    ).toBeDefined();
  });

  it("states the cancel terms in trial details", async () => {
    render(
      <SubscriptionPlans
        plans={PLANS}
        country={COUNTRIES[0]}
        billing="monthly"
        onBillingChange={vi.fn()}
        trialOffer={trialOffer}
        onStartTrial={vi.fn()}
        onSelectPlan={vi.fn()}
        isUpdatingTier={false}
        isStartingTrial={false}
        trialError={null}
      />,
    );
    const pro = within(screen.getByRole("region", { name: "Pro plan" }));
    fireEvent.click(pro.getByRole("button", { name: "Trial details" }));
    const details = await screen.findByRole("dialog");
    expect(
      within(details).getByText(
        "Trial usage is limited. If you cancel, you keep access until the trial ends. You can manage or cancel your plan in billing.",
      ),
    ).toBeDefined();
  });
});
