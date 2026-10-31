import { getGetBrainDumpPlanMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import { server } from "@/mocks/mock-server";
import {
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
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
  NO_PAYWALL_STEPS,
  type Step,
  useOnboardingWizardStore,
} from "../../store";
import { PlanTrialStep } from "../SubscriptionStep/PlanTrialStep";

// One capture for both clients: a trial offer would report through
// @posthog/react, the paywall and the plan through posthog-js.
const posthog = vi.hoisted(() => ({
  __loaded: true,
  is_capturing: () => true,
  capture: vi.fn(),
}));
vi.mock("@posthog/react", () => ({
  usePostHog: () => posthog,
}));
vi.mock("posthog-js", () => ({ default: posthog }));

const PLAN_STEP = NO_PAYWALL_STEPS.subscription as Step;

const readyPlan: PlanResponse = {
  ready: true,
  source: "fallback",
  skipped_dump: false,
  heard: null,
  team: [{ template_id: "maria", name: "Maria", role: "Assistant" }],
  tasks: [
    {
      id: "task-1",
      title: "Draft this week's update",
      prompt: "Draft this week's update",
      expert_template_id: "maria",
    },
  ],
};

beforeEach(() => {
  posthog.capture.mockClear();
  sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().setSteps(NO_PAYWALL_STEPS);
  useOnboardingWizardStore.getState().goToStep(PLAN_STEP);
  server.use(getGetBrainDumpPlanMockHandler200(readyPlan));
});

afterEach(() => {
  setTrialUser(null);
});

function eventsNamed(name: string) {
  return posthog.capture.mock.calls.filter(([event]) => event === name);
}

test("shows the plan on its own, and Continue moves the wizard on", async () => {
  render(<PlanTrialStep hasPaywall={false} />);

  expect(
    screen.getByRole("region", { name: "Here's what your team does first" }),
  ).toBeDefined();
  expect(
    screen.getByRole("heading", {
      level: 1,
      name: "Here's what your team does first",
    }),
  ).toBeDefined();
  expect(await screen.findByText("Draft this week's update")).toBeDefined();

  fireEvent.click(screen.getByRole("button", { name: "Continue" }));

  const wizard = useOnboardingWizardStore.getState();
  expect(wizard.currentStep).toBe(PLAN_STEP + 1);
  expect(wizard.completedSteps).toContain("subscription");
});

test("names when the team joins: when the user continues", async () => {
  render(<PlanTrialStep hasPaywall={false} />);

  expect(
    await screen.findByText(
      "Maria joins your team when you continue. Nothing runs until you pick a task.",
    ),
  ).toBeDefined();
});

test("shows no paywall: no paywall_viewed, no trial, and no checkout request", async () => {
  // Signed in with a trial on offer, so a paywall here would ask for it.
  setTrialUser();
  const requests = vi.fn();
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(() => {
      requests("trial status");
      return trialResponse({
        eligible: true,
        active: false,
        status: null,
        offer: trialOffer,
      });
    }),
    http.post("*/api/credits/trial", () => {
      requests("trial checkout");
      return HttpResponse.json({ url: "https://checkout.stripe.com/trial" });
    }),
    http.post("*/api/credits/subscription", () => {
      requests("tier");
      return HttpResponse.json({ url: null });
    }),
  );

  render(<PlanTrialStep hasPaywall={false} />);
  await screen.findByText("Draft this week's update");
  // The plan loaded and reported itself, so the step's mount requests and
  // effects have all run.
  await waitFor(() => expect(eventsNamed("plan_viewed")).toHaveLength(1));
  fireEvent.click(screen.getByRole("button", { name: "Continue" }));

  expect(useOnboardingWizardStore.getState().currentStep).toBe(PLAN_STEP + 1);
  expect(requests).not.toHaveBeenCalled();
  expect(eventsNamed("paywall_viewed")).toEqual([]);
  expect(eventsNamed("trial_offer_viewed")).toEqual([]);
  expect(useOnboardingWizardStore.getState().selectedPlan).toBeNull();
  expect(
    screen.queryByRole("button", { name: "Start my free trial" }),
  ).toBeNull();
  expect(
    screen.queryByRole("button", { name: "Continue to checkout" }),
  ).toBeNull();
});
