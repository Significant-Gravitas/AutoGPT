import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { act, render, screen, within } from "@/tests/integrations/test-utils";
import { HomeRecap } from "../components/HomeRecap/HomeRecap";
import { makeDashboard } from "./heldItems";

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { user_metadata: { preferred_name: "Abhi" } } }),
}));

let taskQueries: URLSearchParams[] = [];

/** Answers like the API: only the tasks whose status the query names. */
function serveTasks(tasks: Task[]) {
  server.use(
    http.get("/api/proxy/api/tasks", ({ request }) => {
      const params = new URL(request.url).searchParams;
      taskQueries.push(params);
      const status = params.get("status");
      return HttpResponse.json({
        tasks: tasks.filter((task) => !status || task.status === status),
      });
    }),
  );
}

beforeEach(() => {
  taskQueries = [];
  const [leo] = getListExpertsResponseMock200();
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
    getListExpertsMockHandler200([{ ...leo, id: "expert-leo", name: "Leo" }]),
  );
});

describe("Home — tasks that need you", () => {
  it("lists the tasks waiting on the user, each with who has it", async () => {
    serveTasks([
      makeTask("task-1", "Pick a venue for the offsite", "NEEDS_INPUT"),
      makeTask("task-2", "Pull churn numbers", "NEEDS_INPUT", "expert-leo"),
      makeTask("task-3", "Draft the launch email", "IN_PROGRESS"),
    ]);

    render(<HomeRecap />);

    const block = await screen.findByRole("region", {
      name: "Tasks that need you",
    });
    const venue = within(block).getByRole("link", {
      name: /Pick a venue for the offsite/,
    });
    expect(venue.getAttribute("href")).toBe("/tasks?task=task-1");
    expect(venue.textContent).toContain("Otto");
    expect(await within(block).findByText(/^Leo ·/)).toBeDefined();
    expect(within(block).getByText("2")).toBeDefined();
    expect(within(block).queryByText("Draft the launch email")).toBeNull();
    expect(taskQueries.at(-1)?.get("status")).toBe("NEEDS_INPUT");
  });

  it("shows nothing when no task needs the user", async () => {
    serveTasks([makeTask("task-3", "Draft the launch email", "IN_PROGRESS")]);

    render(<HomeRecap />);

    await screen.findByRole("heading", { name: "Your recap" });
    await vi.waitFor(() => expect(taskQueries.length).toBeGreaterThan(0));
    // Lets the answer render, so the absence is not just "not loaded yet".
    await act(async () => {});
    expect(
      screen.queryByRole("region", { name: "Tasks that need you" }),
    ).toBeNull();
    expect(screen.queryByText("Draft the launch email")).toBeNull();
  });
});
