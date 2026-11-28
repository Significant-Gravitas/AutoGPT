import { getGetV1GetUserTimezoneMockHandler } from "@/app/api/__generated__/endpoints/auth/auth.msw";
import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, test } from "vitest";
import { SelectedScheduleView } from "../components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/SelectedScheduleView";

// A schedule whose runs fail the workflow's checks is kept, and each run is
// skipped (#17245). Its view says why, and what to do about it.

function showSchedule(runsSkipped: unknown) {
  const agent = getGetV2GetLibraryAgentResponseMock({
    id: "agent-id",
    graph_id: "graph-id",
    graph_version: 1,
    name: "Daily digest",
    input_schema: { properties: {} },
    credentials_input_schema: { properties: {} },
  });
  server.use(
    getGetV1GetUserTimezoneMockHandler({ timezone: "America/New_York" }),
    http.get("/api/proxy/api/graphs/:graphId/schedules", ({ params }) =>
      HttpResponse.json([
        {
          id: "schedule-id",
          name: "Morning run",
          user_id: "user-id",
          graph_id: params.graphId,
          graph_version: 1,
          cron: "0 9 * * *",
          timezone: "UTC",
          next_run_time: "2026-11-29T09:00:00Z",
          input_data: {},
          input_credentials: {},
          runs_skipped: runsSkipped,
        },
      ]),
    ),
  );
  render(<SelectedScheduleView agent={agent} scheduleId="schedule-id" />);
}

describe("a schedule that skips its runs", () => {
  test.each([
    [
      "key_removed",
      "The key it used was removed. Schedule the workflow again with another key, then delete this schedule.",
    ],
    [
      "key_unusable",
      "It can’t use one of its keys. Schedule the workflow again with a key that works, then delete this schedule.",
    ],
    [
      "inputs_invalid",
      "Its inputs no longer pass the workflow’s checks. Schedule the workflow again with inputs that do, then delete this schedule.",
    ],
    [
      "step_disabled",
      "One of its steps is switched off. It runs again once the step is back on. To run it before then, replace the step and schedule the workflow again.",
    ],
  ])("says why for %s, and since when", async (problem, text) => {
    showSchedule({
      problem,
      fingerprint: "f1",
      since: "2026-11-28T09:00:00Z",
    });

    expect(
      await screen.findByText("This schedule is skipping its runs"),
    ).toBeDefined();
    expect(screen.getByText(text)).toBeDefined();
    expect(
      await screen.findByText("Skipping since November 28, 2026 at 04:00"),
    ).toBeDefined();
  });

  test("a schedule that runs shows no notice", async () => {
    showSchedule(null);

    expect(await screen.findByText("Next run")).toBeDefined();
    expect(screen.queryByText("This schedule is skipping its runs")).toBeNull();
  });
});
