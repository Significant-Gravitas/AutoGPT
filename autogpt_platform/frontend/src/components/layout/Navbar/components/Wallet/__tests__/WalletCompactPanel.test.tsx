import { render, screen } from "@/tests/integrations/test-utils";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { WalletCompactPanel } from "../components/WalletCompactPanel";
import { getEarnGroups, getTaskGroups, TaskGroup } from "../helpers";

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
        id: "VISIT_COPILOT",
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
  "VISIT_COPILOT",
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
          key: "VISIT_COPILOT",
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
  it("shows the balance and one header row per group", () => {
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        onAddCredits={() => {}}
      />,
    );

    expect(screen.getByText("Automation credits")).toBeDefined();
    expect(screen.getByText("$9.79")).toBeDefined();
    expect(screen.getByText("Earn credits")).toBeDefined();
    expect(screen.getByText("First Wins · 2 of 2")).toBeDefined();
    expect(screen.getByText("Done")).toBeDefined();
    expect(screen.getByText("Consistency Challenge · 0 of 2")).toBeDefined();
  });

  it("hides a completed group's tasks until its header is expanded", async () => {
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        onAddCredits={() => {}}
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
        onAddCredits={() => {}}
      />,
    );

    expect(screen.getByText("Schedule your first agent")).toBeDefined();
    expect(screen.getByText("$2.00")).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: /Consistency Challenge/ }),
    );

    expect(screen.queryByText("Schedule your first agent")).toBeNull();
  });

  it("collapses a group once its last task is completed", () => {
    const { rerender } = render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={["SCHEDULE_AGENT"]}
        formattedCredits="$9.79"
        onAddCredits={() => {}}
      />,
    );

    expect(screen.getByText("Run agents 3 days in a row")).toBeDefined();

    rerender(
      <WalletCompactPanel
        groups={groups}
        completedSteps={["SCHEDULE_AGENT", "RUN_3_DAYS"]}
        formattedCredits="$9.79"
        onAddCredits={() => {}}
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
        onAddCredits={onAddCredits}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Add credits" }));

    expect(onAddCredits).toHaveBeenCalledTimes(1);
  });
});
