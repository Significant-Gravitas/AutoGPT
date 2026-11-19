import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getListTasksMockHandler200,
  getListTaskSuggestionsMockHandler200,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { useWorkAreaStore } from "../../../workArea";
import { BriefTab } from "../BriefTab";

const HOUR = 3_600_000;

function serve(tasks: Task[]) {
  server.use(
    getListExpertsMockHandler200([]),
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
    getListTasksMockHandler200({ tasks }),
    getListTaskSuggestionsMockHandler200({ tasks: [] }),
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
  );
}

beforeEach(() => {
  useWorkAreaStore.getState().reset();
});

describe("Brief — This week", () => {
  it("stays away on a first day with nothing to count", async () => {
    serve([]);
    render(<BriefTab />);

    expect(await screen.findByText(/./, { selector: "p" })).toBeDefined();
    expect(screen.queryByRole("region", { name: "This week" })).toBeNull();
    expect(screen.queryByText(/0m/)).toBeNull();
    expect(screen.queryByText("Tasks finished")).toBeNull();
  });

  it("shows only the figures there is something to count for", async () => {
    const now = Date.now();
    serve([
      {
        ...makeTask("task-1", "Draft the post", "DONE"),
        started_at: new Date(now - 2 * HOUR),
        completed_at: new Date(now - HOUR),
        updated_at: new Date(now - HOUR),
      },
    ]);
    render(<BriefTab />);

    const week = await screen.findByRole("region", { name: "This week" });
    expect(within(week).getByText("Tasks finished")).toBeDefined();
    expect(within(week).getByText("1")).toBeDefined();
    expect(within(week).getByText("Work handed off")).toBeDefined();
    expect(within(week).getAllByText("1h")).toHaveLength(2);
    expect(within(week).queryByText("Chats")).toBeNull();
  });
});
