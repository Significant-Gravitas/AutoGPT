import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const authState = vi.hoisted(() => ({ userId: "user-1" }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: authState.userId } }),
}));

vi.mock("./SidebarItemCard", () => ({
  SidebarItemCard: ({ actions }: { actions: ReactNode }) => (
    <div>{actions}</div>
  ),
}));

vi.mock("./ScheduleActionsDropdown", () => ({
  ScheduleActionsDropdown: ({ canDelete }: { canDelete: boolean }) => (
    <div data-testid="schedule-can-delete">{String(canDelete)}</div>
  ),
}));

import { ScheduleListItem } from "./ScheduleListItem";

const agent = { graph_id: "graph-1" } as LibraryAgent;

function schedule(userId: string) {
  return {
    id: "schedule-1",
    name: "Schedule",
    next_run_time: "2026-08-28T09:00:00Z",
    user_id: userId,
  } as GraphExecutionJobInfo;
}

beforeEach(() => {
  authState.userId = "user-1";
});

describe("ScheduleListItem", () => {
  it("allows the schedule owner to delete", () => {
    render(<ScheduleListItem agent={agent} schedule={schedule("user-1")} />);

    expect(screen.getByTestId("schedule-can-delete").textContent).toBe("true");
  });

  it("hides deletion for another team member's schedule", () => {
    render(<ScheduleListItem agent={agent} schedule={schedule("user-2")} />);

    expect(screen.getByTestId("schedule-can-delete").textContent).toBe("false");
  });
});
