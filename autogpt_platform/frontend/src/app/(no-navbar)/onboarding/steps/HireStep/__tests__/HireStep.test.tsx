import { getGetBrainDumpRecommendedExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { getHireExpertMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertRecommendations } from "@/app/api/__generated__/models/expertRecommendations";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

const THREE_TEAM: ExpertRecommendations = {
  ...TEAM,
  experts: [
    ...TEAM.experts!,
    {
      template_id: "tpl-sam",
      name: "Sam",
      role: "Support",
      avatar_url: null,
      reason: "Tickets pile up over the weekend.",
      workflow_names: [],
    },
  ],
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

// Records every hire's template id; `failing` answers 503 until removed.
function mockHires(failing = new Set<string>(), gate?: Promise<void>) {
  const hired: string[] = [];
  server.use(
    http.post("*/api/experts", async ({ request }) => {
      const { template_id } = (await request.json()) as {
        template_id: string;
      };
      hired.push(template_id);
      await gate;
      return failing.has(template_id)
        ? HttpResponse.json({ detail: "unavailable" }, { status: 503 })
        : HttpResponse.json({ expert: { id: `exp-${template_id}` } });
    }),
  );
  return hired;
}

function events(name: string) {
  return capture.mock.calls.filter(([event]) => event === name);
}

beforeEach(() => {
  capture.mockReset();
  toast.mockReset();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().goToStep(5);
});

describe("HireStep — the team", () => {
  it("renders Otto's read, a card per expert and a Hire button each", async () => {
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
      team_source: "llm",
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
    expect(screen.getByTestId("hire-step-diagnosis").textContent).toBe(
      "Hard to say what eats your week yet.",
    );
    expect(screen.queryAllByTestId("hire-step-expert")).toHaveLength(0);
    expect(screen.getByRole("button", { name: "Continue" })).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "I'll add members later" }),
    ).toBeNull();
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
    expect(hires).toEqual([
      { template_id: "tpl-maria", surface: "onboarding" },
    ]);
    expect(useOnboardingWizardStore.getState().hiredTemplateIds).toEqual([
      "tpl-maria",
    ]);
    // One hire down, one still open.
    expect(screen.getAllByRole("button", { name: "Hire" })).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Hire Max too" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Continue" })).toBeDefined();
    // One hire_started per click; the hire itself is the backend's
    // expert_hired, so the browser sends no completion event.
    expect(
      capture.mock.calls.filter(([event]) => event === "hire_started"),
    ).toEqual([
      ["hire_started", { template_id: "tpl-maria", surface: "onboarding" }],
    ]);
    expect(capture).not.toHaveBeenCalledWith(
      "onboarding_expert_hired",
      expect.anything(),
    );
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
    await userEvent.click(
      await screen.findByRole("button", { name: "Continue" }),
    );

    expect(useOnboardingWizardStore.getState().currentStep).toBe(6);
    expect(capture).toHaveBeenCalledWith("hire_step_continued", {
      hired: 1,
      recommended: 2,
      via: "continue",
    });
  });
});

describe("HireStep — hiring the whole team", () => {
  it.each([
    {
      team: TEAM,
      hired: [],
      primary: "Hire all 2",
      skip: "I'll add members later",
    },
    {
      team: { ...TEAM, experts: TEAM.experts!.slice(0, 1) },
      hired: [],
      primary: "Hire Maria",
      skip: "I'll add members later",
    },
    {
      team: THREE_TEAM,
      hired: ["tpl-maria"],
      primary: "Hire the other 2",
      skip: "Continue",
    },
    {
      team: THREE_TEAM,
      hired: ["tpl-maria", "tpl-max"],
      primary: "Hire Sam too",
      skip: "Continue",
    },
    {
      team: THREE_TEAM,
      hired: ["tpl-maria", "tpl-max", "tpl-sam"],
      primary: "Continue",
      skip: null,
    },
  ])(
    "offers '$primary' with $hired.length hired",
    async ({ team, hired, primary, skip }) => {
      mockTeam(team);
      hired.forEach((id) => useOnboardingWizardStore.getState().markHired(id));
      render(<HireStep />);

      expect(
        await screen.findByRole("button", { name: primary }),
      ).toBeDefined();
      const skips = ["I'll add members later", "Continue"].filter(
        (name) => name !== primary && screen.queryByRole("button", { name }),
      );
      expect(skips).toEqual(skip ? [skip] : []);
    },
  );

  it("hires every recommended expert in one click, then moves on", async () => {
    mockTeam(THREE_TEAM);
    const hired = mockHires();
    render(<HireStep />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire all 3" }),
    );

    expect(await screen.findAllByText("Hired")).toHaveLength(3);
    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(6),
    );
    expect(hired).toEqual(["tpl-maria", "tpl-max", "tpl-sam"]);
    expect(useOnboardingWizardStore.getState().hiredTemplateIds).toEqual([
      "tpl-maria",
      "tpl-max",
      "tpl-sam",
    ]);
    expect(events("hire_all_clicked")).toEqual([
      [
        "hire_all_clicked",
        { count: 3, already_hired: 0, recommended: 3, retry: false },
      ],
    ]);
    expect(events("hire_started")).toHaveLength(3);
    expect(events("expert_recommendation_clicked")).toEqual([]);
    expect(events("hire_step_continued")).toEqual([
      ["hire_step_continued", { hired: 3, recommended: 3, via: "auto" }],
    ]);
  });

  it("counts who is left in as cards are left out, down to the skip alone", async () => {
    mockTeam(THREE_TEAM);
    render(<HireStep />);

    await screen.findByRole("button", { name: "Hire all 3" });
    const max = screen.getByRole("button", { name: "Include Max" });
    expect(max.getAttribute("aria-pressed")).toBe("true");

    await userEvent.click(max);
    expect(max.getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByRole("button", { name: "Hire 2" })).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Include Sam" }));
    expect(screen.getByRole("button", { name: "Hire Maria" })).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: "Include Maria" }),
    );
    const stepActions = screen
      .getAllByRole("button")
      .filter(
        (button) =>
          !button.getAttribute("aria-label")?.startsWith("Include") &&
          button.textContent !== "Hire",
      )
      .map((button) => button.textContent);
    expect(stepActions).toEqual(["I'll add members later"]);

    await userEvent.click(screen.getByRole("button", { name: "Include Sam" }));
    expect(screen.getByRole("button", { name: "Hire Sam" })).toBeDefined();
  });

  it("does not hire an expert the user left out", async () => {
    mockTeam(THREE_TEAM);
    const hired = mockHires();
    render(<HireStep />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Include Max" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Hire 2" }));

    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(6),
    );
    expect(hired).toEqual(["tpl-maria", "tpl-sam"]);
    expect(useOnboardingWizardStore.getState().hiredTemplateIds).toEqual([
      "tpl-maria",
      "tpl-sam",
    ]);
    expect(events("hire_all_clicked")).toEqual([
      [
        "hire_all_clicked",
        { count: 2, already_hired: 0, recommended: 3, retry: false },
      ],
    ]);
  });

  it("hides the skip and holds Back from the click until the step moves on", async () => {
    mockTeam(THREE_TEAM);
    let release!: () => void;
    mockHires(new Set(), new Promise<void>((resolve) => (release = resolve)));
    // Back is hidden while the step is busy; a Back during the advance beat
    // would land the timer's nextStep on the previous step.
    const busyChangesOnHireStep: boolean[] = [];
    const unsubscribe = useOnboardingWizardStore.subscribe((state, prev) => {
      if (state.currentStep === 5 && state.isStepBusy !== prev.isStepBusy) {
        busyChangesOnHireStep.push(state.isStepBusy);
      }
    });
    render(<HireStep />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire all 3" }),
    );

    expect(
      await screen.findByRole("button", { name: "Hiring…" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "I'll add members later" }),
    ).toBeNull();
    release();
    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(6),
    );
    unsubscribe();
    expect(busyChangesOnHireStep).toEqual([true]);
  });

  it.each([
    { when: "before its hires land", landFirst: false },
    { when: "during the advance beat", landFirst: true },
  ])(
    "does not move the wizard on when the step is left $when",
    async ({ landFirst }) => {
      mockTeam(THREE_TEAM);
      let release!: () => void;
      mockHires(new Set(), new Promise<void>((resolve) => (release = resolve)));
      const { unmount } = render(<HireStep />);

      await userEvent.click(
        await screen.findByRole("button", { name: "Hire all 3" }),
      );
      await screen.findByRole("button", { name: "Hiring…" });
      if (landFirst) {
        release();
        expect(await screen.findAllByText("Hired")).toHaveLength(3);
      }
      // Browser navigation unmounts the step; Back being hidden does not stop it.
      unmount();
      render(<HireStep />);
      await screen.findAllByTestId("hire-step-expert");
      release();

      await waitFor(() =>
        expect(
          useOnboardingWizardStore.getState().hiredTemplateIds,
        ).toHaveLength(3),
      );
      await new Promise((resolve) => setTimeout(resolve, 800));
      expect(useOnboardingWizardStore.getState().currentStep).toBe(5);
      expect(events("hire_step_continued")).toEqual([]);
    },
  );

  it("keeps the hires that landed, names the one that did not, and offers to try it again", async () => {
    mockTeam(THREE_TEAM);
    const failing = new Set(["tpl-sam"]);
    const hired = mockHires(failing);
    render(<HireStep />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire all 3" }),
    );

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({
        title: "Couldn't hire Sam",
        description:
          "Something went wrong. Try again, or continue without them.",
        variant: "destructive",
      }),
    );
    expect(screen.getAllByText("Hired")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Continue" })).toBeDefined();
    // A failure never auto-advances, however long the user looks at it.
    await new Promise((resolve) => setTimeout(resolve, 800));
    expect(useOnboardingWizardStore.getState().currentStep).toBe(5);

    failing.clear();
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(6),
    );
    expect(hired).toEqual(["tpl-maria", "tpl-max", "tpl-sam", "tpl-sam"]);
    expect(events("hire_all_clicked")[1]).toEqual([
      "hire_all_clicked",
      { count: 1, already_hired: 2, recommended: 3, retry: true },
    ]);
  });

  it("says the team could not be hired when every hire fails", async () => {
    mockTeam(THREE_TEAM);
    mockHires(new Set(["tpl-maria", "tpl-max", "tpl-sam"]));
    render(<HireStep />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire all 3" }),
    );

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({
        title: "Couldn't hire your team",
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      }),
    );
    expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
    expect(
      screen.getByRole("button", { name: "I'll add members later" }),
    ).toBeDefined();
    expect(screen.queryByText("Hired")).toBeNull();
  });

  it("starts one batch and moves on once on a double click", async () => {
    mockTeam(THREE_TEAM);
    let release!: () => void;
    const hired = mockHires(
      new Set(),
      new Promise<void>((resolve) => (release = resolve)),
    );
    render(<HireStep />);

    await userEvent.dblClick(
      await screen.findByRole("button", { name: "Hire all 3" }),
    );
    release();

    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(6),
    );
    // Past the advance beat: a second nextStep would skip the next step.
    await new Promise((resolve) => setTimeout(resolve, 800));
    expect(useOnboardingWizardStore.getState().currentStep).toBe(6);
    expect(hired).toEqual(["tpl-maria", "tpl-max", "tpl-sam"]);
    expect(events("hire_step_continued")).toHaveLength(1);
  });

  it("still lets the user skip without hiring anyone", async () => {
    mockTeam(TEAM);
    const hired = mockHires();
    render(<HireStep />);

    await userEvent.click(
      await screen.findByRole("button", { name: "I'll add members later" }),
    );

    expect(useOnboardingWizardStore.getState().currentStep).toBe(6);
    expect(hired).toEqual([]);
    expect(events("hire_step_continued")).toEqual([
      ["hire_step_continued", { hired: 0, recommended: 2, via: "later" }],
    ]);
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
    expect(screen.getByTestId("hire-step-primary-pending")).toBeDefined();
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
      "Nothing to hire yet — you can add members later",
      {},
      { timeout: 22000 },
    ),
  ).toBeDefined();
  expect(screen.queryByTestId("hire-step-diagnosis")).toBeNull();
  expect(calls).toBeGreaterThan(1);
  const stoppedAt = calls;
  await new Promise((resolve) => setTimeout(resolve, 3000));
  expect(calls).toBe(stoppedAt);
}, 26000);

afterEach(() => {
  vi.unstubAllEnvs();
});
