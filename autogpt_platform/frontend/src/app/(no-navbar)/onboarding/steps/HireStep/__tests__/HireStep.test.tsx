import { getGetBrainDumpRecommendedExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { getHireExpertMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertRecommendations } from "@/app/api/__generated__/models/expertRecommendations";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useOnboardingWizardStore } from "../../../store";
import { HireStep } from "../HireStep";

const { capture, toast } = vi.hoisted(() => ({
  capture: vi.fn(),
  toast: vi.fn(),
}));
vi.mock("posthog-js", () => ({ default: { capture } }));
// Spread the real module: OnboardingProvider in test-utils calls useToast.
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

const TEAM: ExpertRecommendations = {
  diagnosis: "You have a marketing problem and a support problem.",
  source: "llm",
  experts: [
    {
      template_id: "tpl-maria",
      name: "Maria",
      role: "Marketing",
      avatar_url: null,
      reason: "Your LinkedIn posts eat a day a week.",
      workflow_names: [
        "Weekly LinkedIn post",
        "Content calendar",
        "Newsletter draft",
        "Never shown",
      ],
    },
    {
      template_id: "tpl-max",
      name: "Max",
      role: "Sales",
      avatar_url: null,
      reason: "Cold outreach never gets sent.",
      workflow_names: [],
    },
  ],
  raise_suggestion: {
    role: "support",
    reason: "Nobody on the roster handles support tickets.",
  },
};

const FALLBACK_TEAM: ExpertRecommendations = {
  ...TEAM,
  source: "fallback",
  raise_suggestion: null,
};

const EMPTY_TEAM: ExpertRecommendations = {
  diagnosis: "Hard to say what eats your week yet.",
  source: "fallback",
  experts: [],
  raise_suggestion: null,
};

function mockTeam(team: ExpertRecommendations | null, ready = true) {
  server.use(getGetBrainDumpRecommendedExpertsMockHandler200({ ready, team }));
}

beforeEach(() => {
  capture.mockReset();
  toast.mockReset();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().goToStep(5);
});

describe("HireStep — the team", () => {
  it("renders AutoPilot's read, a card per expert and a Hire button each", async () => {
    mockTeam(TEAM);
    render(<HireStep />);

    expect(
      await screen.findByText("Based on what you told me, here's who I'd hire"),
    ).toBeDefined();
    expect(screen.getAllByTestId("hire-step-expert")).toHaveLength(2);
    expect(screen.getByText("Maria")).toBeDefined();
    expect(
      screen.getByText("Your LinkedIn posts eat a day a week."),
    ).toBeDefined();
    // The card is name, role and reason; workflow names stay off it.
    expect(screen.queryByText("Weekly LinkedIn post")).toBeNull();
    expect(screen.getAllByRole("button", { name: "Hire" })).toHaveLength(2);
    expect(
      screen.getByRole("button", { name: "I'll add members later" }),
    ).toBeDefined();
  });

  it("says the team came from the wizard answers when the dump was not read", async () => {
    mockTeam(FALLBACK_TEAM);
    render(<HireStep />);

    expect(
      await screen.findByText("Based on your role, here's who I'd hire first"),
    ).toBeDefined();
    expect(screen.queryByTestId("hire-step-raise-note")).toBeNull();
  });

  it("reports every card it showed exactly once across polls", async () => {
    mockTeam(TEAM);
    render(<HireStep />);
    await screen.findAllByTestId("hire-step-expert");

    expect(capture).toHaveBeenCalledWith("expert_recommended", {
      template_id: "tpl-maria",
      position: 0,
      source: "llm",
    });
    expect(
      capture.mock.calls.filter(([event]) => event === "expert_recommended"),
    ).toHaveLength(2);
  });

  it("names the gap the roster cannot fill without leaving the wizard", async () => {
    mockTeam(TEAM);
    render(<HireStep />);

    expect(await screen.findByTestId("hire-step-raise-note")).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("still offers the way forward with no experts to hire", async () => {
    mockTeam(EMPTY_TEAM);
    render(<HireStep />);

    expect(
      await screen.findByText("Here's my read on your team"),
    ).toBeDefined();
    expect(screen.queryAllByTestId("hire-step-expert")).toHaveLength(0);
    expect(
      screen.getByRole("button", { name: "I'll add members later" }),
    ).toBeDefined();
  });
});

describe("HireStep — hiring", () => {
  it("hires the clicked template, marks the card and remembers it in the store", async () => {
    mockTeam(TEAM);
    const hires: unknown[] = [];
    server.use(
      http.post("*/api/experts", async ({ request }) => {
        hires.push(await request.json());
        return HttpResponse.json({
          expert: { id: "exp-1", name: "Maria" },
          failed_preloads: [],
        });
      }),
    );
    render(<HireStep />);

    await userEvent.click(
      (await screen.findAllByRole("button", { name: "Hire" }))[0],
    );

    await waitFor(() => {
      expect(screen.getByText("Hired")).toBeDefined();
    });
    expect(hires).toEqual([{ template_id: "tpl-maria" }]);
    expect(useOnboardingWizardStore.getState().hiredTemplateIds).toEqual([
      "tpl-maria",
    ]);
    // One hire down, one still open.
    expect(screen.getAllByRole("button", { name: "Hire" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Next" })).toBeDefined();
    expect(capture).toHaveBeenCalledWith("hire_started", {
      template_id: "tpl-maria",
      source: "onboarding_hire_step",
    });
    expect(capture).toHaveBeenCalledWith("onboarding_expert_hired", {
      template_id: "tpl-maria",
      position: 0,
    });
  });

  it("shows templates hired on an earlier visit as hired", async () => {
    mockTeam(TEAM);
    useOnboardingWizardStore.getState().markHired("tpl-max");
    render(<HireStep />);

    await screen.findAllByTestId("hire-step-expert");
    expect(screen.getByText("Hired")).toBeDefined();
    expect(screen.getAllByRole("button", { name: "Hire" })).toHaveLength(1);
  });

  it("keeps the card open and says so when the hire fails", async () => {
    mockTeam(TEAM);
    server.use(
      http.post("*/api/experts", () =>
        HttpResponse.json({ detail: "nope" }, { status: 503 }),
      ),
    );
    render(<HireStep />);

    await userEvent.click(
      (await screen.findAllByRole("button", { name: "Hire" }))[0],
    );

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't hire Maria",
          variant: "destructive",
        }),
      );
    });
    expect(screen.getAllByRole("button", { name: "Hire" })).toHaveLength(2);
    expect(useOnboardingWizardStore.getState().hiredTemplateIds).toEqual([]);
  });

  it("advances the wizard and reports how many were hired", async () => {
    mockTeam(TEAM);
    server.use(getHireExpertMockHandler200());
    render(<HireStep />);

    await userEvent.click(
      (await screen.findAllByRole("button", { name: "Hire" }))[0],
    );
    await screen.findByRole("button", { name: "Next" });
    await userEvent.click(screen.getByRole("button", { name: "Next" }));

    expect(useOnboardingWizardStore.getState().currentStep).toBe(6);
    expect(capture).toHaveBeenCalledWith("hire_step_continued", {
      hired: 1,
      recommended: 2,
    });
  });
});

describe("HireStep — while the job is still running", () => {
  it("shows skeleton cards and the way out, then the team when it lands", async () => {
    let ready = false;
    server.use(
      http.get("*/api/onboarding/brain-dump/recommended-experts", () =>
        HttpResponse.json(
          ready ? { ready: true, team: TEAM } : { ready: false },
        ),
      ),
    );
    render(<HireStep />);

    expect(await screen.findAllByTestId("hire-step-pending")).not.toHaveLength(
      0,
    );
    expect(screen.getByText("Thinking about who you'd need…")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "I'll add members later" }),
    ).toBeDefined();

    ready = true;
    expect(
      await screen.findByText("Maria", {}, { timeout: 5_000 }),
    ).toBeDefined();
    expect(screen.queryAllByTestId("hire-step-pending")).toHaveLength(0);
  });
});

it.each([408, 429, 503])(
  "recovers team recommendations after a transient %s response",
  async (status) => {
    let calls = 0;
    server.use(
      http.get("*/api/onboarding/brain-dump/recommended-experts", () => {
        calls += 1;
        return calls === 1
          ? HttpResponse.json({ detail: "try again" }, { status })
          : HttpResponse.json({ ready: true, team: TEAM });
      }),
    );
    render(<HireStep />);
    await waitFor(() => expect(calls).toBe(1));
    expect(screen.queryAllByTestId("hire-step-pending")).not.toHaveLength(0);
    expect(
      await screen.findByText("Maria", {}, { timeout: 5000 }),
    ).toBeDefined();
    expect(calls).toBe(2);
  },
);

it("stops waiting after the deadline if transient errors persist", async () => {
  let calls = 0;
  server.use(
    http.get("*/api/onboarding/brain-dump/recommended-experts", () => {
      calls += 1;
      return HttpResponse.json({ detail: "unavailable" }, { status: 503 });
    }),
  );
  render(<HireStep />);
  expect(
    await screen.findByText(
      "Here's my read on your team",
      {},
      { timeout: 22000 },
    ),
  ).toBeDefined();
  expect(calls).toBeGreaterThan(1);
  const stoppedAt = calls;
  await new Promise((resolve) => setTimeout(resolve, 3000));
  expect(calls).toBe(stoppedAt);
}, 26000);
