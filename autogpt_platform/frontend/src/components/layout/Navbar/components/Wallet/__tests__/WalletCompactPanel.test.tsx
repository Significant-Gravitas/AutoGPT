import { render, screen } from "@/tests/integrations/test-utils";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { WalletCompactPanel } from "../components/WalletCompactPanel";
import {
  getEarnGroups,
  getTaskGroups,
  getTrialPlanLine,
  TaskGroup,
} from "../helpers";

// Noon on Sep 17 in the runner's own time zone: the trial date is shown in the
// viewer's time zone, so a fixed UTC instant lands on Sep 18 east of UTC+9.
const SEP_17_LOCAL_NOON = new Date(2030, 8, 17, 12);

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: vi.fn(() => true) };
});

const groups: TaskGroup[] = [
  {
    name: "First Wins",
    details: "",
    tasks: [
      {
        id: "ONBOARDING_COMPLETE",
        name: "Complete onboarding",
        amount: 3,
        details: "",
      },
      {
        id: "MARKETPLACE_ADD_AGENT",
        name: "Get an agent from the marketplace",
        amount: 1,
        details: "",
      },
    ],
  },
  {
    name: "Consistency Challenge",
    details: "",
    tasks: [
      {
        id: "SCHEDULE_AGENT",
        name: "Schedule your first agent",
        amount: 1,
        details: "",
      },
      {
        id: "RUN_3_DAYS",
        name: "Run agents 3 days in a row",
        amount: 2,
        details: "",
      },
    ],
  },
];

const completedSteps: OnboardingStep[] = [
  "ONBOARDING_COMPLETE",
  "MARKETPLACE_ADD_AGENT",
];

describe("getEarnGroups", () => {
  it("marks a fully completed group done and collapsed by default", () => {
    const earnGroups = getEarnGroups(groups, completedSteps);

    expect(earnGroups[0]).toEqual({
      key: "First Wins",
      label: "First Wins · 2 of 2",
      done: true,
      amount: 0,
      defaultOpen: false,
      rows: [
        {
          key: "ONBOARDING_COMPLETE",
          label: "Complete onboarding",
          done: true,
          amount: 3,
        },
        {
          key: "MARKETPLACE_ADD_AGENT",
          label: "Get an agent from the marketplace",
          done: true,
          amount: 1,
        },
      ],
    });
  });

  it("keeps an in-progress group expanded and sums the remaining rewards", () => {
    const earnGroups = getEarnGroups(groups, ["SCHEDULE_AGENT"]);

    expect(earnGroups[1].label).toBe("Consistency Challenge · 1 of 2");
    expect(earnGroups[1].done).toBe(false);
    expect(earnGroups[1].defaultOpen).toBe(true);
    expect(earnGroups[1].amount).toBe(2);
    expect(earnGroups[1].rows.map((row) => row.done)).toEqual([true, false]);
  });

  it("treats a missing completedSteps list as nothing claimed", () => {
    const earnGroups = getEarnGroups(groups, undefined);

    expect(earnGroups.every((group) => !group.done)).toBe(true);
    expect(
      earnGroups.flatMap((group) => group.rows).every((row) => !row.done),
    ).toBe(true);
  });

  it("keeps every real onboarding task reachable", () => {
    const realGroups = getTaskGroups(null);
    const earnGroups = getEarnGroups(realGroups, []);

    expect(earnGroups.flatMap((group) => group.rows)).toHaveLength(
      realGroups.reduce((total, group) => total + group.tasks.length, 0),
    );
  });
});

describe("WalletCompactPanel", () => {
  it("shows onboarding groups and unfinished tasks as soon as the wallet opens", () => {
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={() => {}}
        onOpenExplainer={() => {}}
        onNavigate={() => {}}
      />,
    );

    expect(screen.getByText("Automation credits")).toBeDefined();
    expect(screen.getByText("$9.79")).toBeDefined();
    expect(screen.getByText("Earn credits")).toBeDefined();
    expect(screen.getByText("First Wins · 2 of 2")).toBeDefined();
    expect(screen.getByText("Done")).toBeDefined();
    expect(screen.getByText("Consistency Challenge · 0 of 2")).toBeDefined();
    expect(screen.getByText("Schedule your first agent")).toBeDefined();
    expect(screen.getByText("Run agents 3 days in a row")).toBeDefined();
  });

  it("clearly identifies onboarding tasks when collapsed and restores them on expand", async () => {
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={() => {}}
        onOpenExplainer={() => {}}
        onNavigate={() => {}}
      />,
    );

    const hideTasks = screen.getByRole("button", {
      name: /Hide onboarding tasks/,
    });
    await userEvent.click(screen.getByRole("button", { name: /First Wins/ }));
    await userEvent.click(
      screen.getByRole("button", { name: /Consistency Challenge/ }),
    );

    expect(hideTasks.getAttribute("aria-expanded")).toBe("true");
    await userEvent.click(hideTasks);

    const showTasks = screen.getByRole("button", {
      name: /Show onboarding tasks/,
    });
    expect(showTasks.getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByText("Earn credits")).toBeDefined();
    expect(screen.getByText("$3.00 available")).toBeDefined();
    expect(screen.queryByRole("button", { name: /First Wins/ })).toBeNull();
    expect(screen.getByText("Automation credits")).toBeDefined();

    await userEvent.click(showTasks);
    expect(
      screen
        .getByRole("button", { name: /First Wins/ })
        .getAttribute("aria-expanded"),
    ).toBe("true");
    expect(
      screen
        .getByRole("button", { name: /Consistency Challenge/ })
        .getAttribute("aria-expanded"),
    ).toBe("false");
    await userEvent.click(
      screen.getByRole("button", { name: /Consistency Challenge/ }),
    );
    expect(screen.getByText("Schedule your first agent")).toBeDefined();
    expect(screen.getByText("Run agents 3 days in a row")).toBeDefined();
  });

  it("hides a completed group's tasks until its header is expanded", async () => {
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={() => {}}
        onOpenExplainer={() => {}}
        onNavigate={() => {}}
      />,
    );

    expect(screen.queryByText("Complete onboarding")).toBeNull();

    await userEvent.click(
      screen.getByRole("button", { name: /First Wins · 2 of 2/ }),
    );

    expect(screen.getByText("Complete onboarding")).toBeDefined();
    expect(screen.getByText("Get an agent from the marketplace")).toBeDefined();
  });

  it("shows an in-progress group's tasks by default and collapses on toggle", async () => {
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={() => {}}
        onOpenExplainer={() => {}}
        onNavigate={() => {}}
      />,
    );

    expect(screen.getByText("Schedule your first agent")).toBeDefined();
    expect(screen.getByText("$2.00")).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: /Consistency Challenge/ }),
    );

    expect(screen.queryByText("Schedule your first agent")).toBeNull();
  });

  it("collapses a group once its last task is completed", async () => {
    const { rerender } = render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={["SCHEDULE_AGENT"]}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={() => {}}
        onOpenExplainer={() => {}}
        onNavigate={() => {}}
      />,
    );

    expect(screen.getByText("Run agents 3 days in a row")).toBeDefined();

    rerender(
      <WalletCompactPanel
        groups={groups}
        completedSteps={["SCHEDULE_AGENT", "RUN_3_DAYS"]}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={() => {}}
        onOpenExplainer={() => {}}
        onNavigate={() => {}}
      />,
    );

    expect(screen.queryByText("Run agents 3 days in a row")).toBeNull();
    expect(screen.getByText("Consistency Challenge · 2 of 2")).toBeDefined();
  });

  it("calls onAddCredits when the Add credits row is clicked", async () => {
    const onAddCredits = vi.fn();
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={onAddCredits}
        onOpenExplainer={() => {}}
        onNavigate={() => {}}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Add credits" }));

    expect(onAddCredits).toHaveBeenCalledTimes(1);
  });

  it("leaves the explanations to the dialog and links to billing settings", async () => {
    const onOpenExplainer = vi.fn();
    const onNavigate = vi.fn();
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        trialNoteID="trial-note"
        onAddCredits={() => {}}
        onOpenExplainer={onOpenExplainer}
        onNavigate={onNavigate}
      />,
    );

    expect(screen.queryByText(/Pays for workflow runs/)).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "How it works" }));
    expect(onOpenExplainer).toHaveBeenCalledTimes(1);

    const billingLink = screen.getByRole("link", { name: /Billing settings/ });
    expect(billingLink.getAttribute("href")).toBe("/settings/billing");
  });
});

describe("getTrialPlanLine", () => {
  const trial = {
    active: true,
    cancel_at_period_end: false,
    ends_at: SEP_17_LOCAL_NOON,
  };

  it("names the day the plan starts during an active trial", () => {
    expect(getTrialPlanLine(trial)).toBe("Your plan starts Sep 17");
  });

  it("falls back to the end of the trial when the date is unknown", () => {
    expect(getTrialPlanLine({ ...trial, ends_at: null })).toBe(
      "Your plan starts when your trial ends",
    );
  });

  it("names the trial's end instead of a plan once the trial is canceled", () => {
    expect(getTrialPlanLine({ ...trial, cancel_at_period_end: true })).toBe(
      "Your trial ends Sep 17",
    );
    expect(
      getTrialPlanLine({ ...trial, cancel_at_period_end: true, ends_at: null }),
    ).toBeNull();
  });

  it("says nothing without an active trial", () => {
    expect(getTrialPlanLine({ ...trial, active: false })).toBeNull();
    expect(getTrialPlanLine({})).toBeNull();
    expect(getTrialPlanLine(undefined)).toBeNull();
  });
});
