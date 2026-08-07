import { render, screen } from "@/tests/integrations/test-utils";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { WalletCompactPanel } from "../components/WalletCompactPanel";
import { getEarnRows, getTaskGroups, TaskGroup } from "../helpers";

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

describe("getEarnRows", () => {
  it("collapses a fully completed group into a single Done row", () => {
    const rows = getEarnRows(groups, completedSteps);

    expect(rows).toEqual([
      {
        key: "First Wins",
        label: "First Wins · 2 of 2",
        done: true,
        amount: 0,
      },
      {
        key: "SCHEDULE_AGENT",
        label: "Schedule your first agent",
        done: false,
        amount: 1,
      },
      {
        key: "RUN_3_DAYS",
        label: "Run agents 3 days in a row",
        done: false,
        amount: 2,
      },
    ]);
  });

  it("lists every task of a group that is still in progress", () => {
    const rows = getEarnRows(groups, ["SCHEDULE_AGENT"]);

    expect(rows.map((row) => row.key)).toEqual([
      "VISIT_COPILOT",
      "MARKETPLACE_ADD_AGENT",
      "SCHEDULE_AGENT",
      "RUN_3_DAYS",
    ]);
    expect(rows.find((row) => row.key === "SCHEDULE_AGENT")?.done).toBe(true);
  });

  it("treats a missing completedSteps list as nothing claimed", () => {
    const rows = getEarnRows(groups, undefined);

    expect(rows.map((row) => row.key)).toEqual([
      "VISIT_COPILOT",
      "MARKETPLACE_ADD_AGENT",
      "SCHEDULE_AGENT",
      "RUN_3_DAYS",
    ]);
    expect(rows.every((row) => !row.done)).toBe(true);
  });

  it("keeps every real onboarding task reachable", () => {
    const realGroups = getTaskGroups(null);
    const rows = getEarnRows(realGroups, []);

    expect(rows).toHaveLength(
      realGroups.reduce((total, group) => total + group.tasks.length, 0),
    );
  });
});

describe("WalletCompactPanel", () => {
  it("shows the balance, the earn list and the completed group as Done", () => {
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
    expect(screen.getByText("Schedule your first agent")).toBeDefined();
    expect(screen.getByText("$2.00")).toBeDefined();
  });

  it("collapses the earn credits list when the accordion is toggled", async () => {
    render(
      <WalletCompactPanel
        groups={groups}
        completedSteps={completedSteps}
        formattedCredits="$9.79"
        onAddCredits={() => {}}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: /Earn credits/ }));

    expect(screen.queryByText("Schedule your first agent")).toBeNull();
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
