import { getGetV1GetUserTimezoneMockHandler } from "@/app/api/__generated__/endpoints/auth/auth.msw";
import {
  getGetV2GetLibraryAgentResponseMock,
  getGetV2ListTriggerAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, test } from "vitest";
import { SelectedScheduleView } from "../components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { SelectedTriggerAgentView } from "../components/NewAgentLibraryView/components/selected-views/SelectedTriggerAgentView/SelectedTriggerAgentView";

describe.each(["schedule", "trigger"] as const)("%s next run", (view) => {
  test.each([
    ["missing", undefined, "Not scheduled"],
    ["null", null, "Not scheduled"],
    ["unparseable", "not-a-date", "Not scheduled"],
    ["empty", "", "Not scheduled"],
    ["object", {}, "Not scheduled"],
    ["boolean", false, "Not scheduled"],
    ["number", 0, "Not scheduled"],
    ["valid", "2024-01-15T12:00:00Z", "January 15, 2024 at 07:00"],
  ] as const)(
    "renders a %s next_run_time from the API",
    async (_, nextRunTime, expected) => {
      const agent = getGetV2GetLibraryAgentResponseMock({
        id: "agent-id",
        graph_id: "graph-id",
        graph_version: 1,
        name: "Scheduled Agent",
        input_schema: { properties: {} },
        credentials_input_schema: { properties: {} },
      });
      const triggerAgent = getGetV2GetLibraryAgentResponseMock({
        id: "trigger-id",
        graph_id: "trigger-graph-id",
        graph_version: 1,
        name: "Scheduled Trigger",
      });

      server.use(
        getGetV1GetUserTimezoneMockHandler({ timezone: "America/New_York" }),
        getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
        http.get("/api/proxy/api/graphs/:graphId/schedules", ({ params }) =>
          HttpResponse.json([
            {
              id: "schedule-id",
              name: "Morning run",
              user_id: "user-id",
              graph_id: params.graphId,
              graph_version: 1,
              cron: "0 8 * * *",
              timezone: "UTC",
              next_run_time: nextRunTime,
              input_data: {},
              input_credentials: {},
            },
          ]),
        ),
      );

      render(
        view === "schedule" ? (
          <SelectedScheduleView agent={agent} scheduleId="schedule-id" />
        ) : (
          <SelectedTriggerAgentView agent={agent} triggerAgentId="trigger-id" />
        ),
      );

      expect(await screen.findByText("Next run")).toBeDefined();
      expect(await screen.findByText(expected)).toBeDefined();
      expect(screen.queryByText(/when retrieving data/i)).toBeNull();
    },
  );
});

describe("SelectedScheduleView timezone labels", () => {
  test("labels Next run with the profile timezone when schedule timezone differs", async () => {
    const agent = getGetV2GetLibraryAgentResponseMock({
      id: "agent-id",
      graph_id: "graph-id",
      graph_version: 1,
      name: "Scheduled Agent",
      input_schema: { properties: {} },
      credentials_input_schema: { properties: {} },
    });

    server.use(
      getGetV1GetUserTimezoneMockHandler({ timezone: "America/Los_Angeles" }),
      http.get("/api/proxy/api/graphs/:graphId/schedules", ({ params }) =>
        HttpResponse.json([
          {
            id: "schedule-id",
            name: "Teig ansetzen",
            user_id: "user-id",
            graph_id: params.graphId,
            graph_version: 1,
            cron: "45 2 * * 1-5",
            timezone: "Asia/Tokyo",
            next_run_time: "2026-11-12T02:45:00+09:00",
            input_data: {},
            input_credentials: {},
          },
        ]),
      ),
    );

    render(<SelectedScheduleView agent={agent} scheduleId="schedule-id" />);

    expect(await screen.findByText("November 11, 2026 at 09:45")).toBeDefined();
    expect(
      await screen.findByText(getTimezoneDisplayName("America/Los_Angeles")),
    ).toBeDefined();
  });

  test("falls back to browser timezone label when profile timezone is not-set", async () => {
    const agent = getGetV2GetLibraryAgentResponseMock({
      id: "agent-id",
      graph_id: "graph-id",
      graph_version: 1,
      name: "Scheduled Agent",
      input_schema: { properties: {} },
      credentials_input_schema: { properties: {} },
    });

    server.use(
      getGetV1GetUserTimezoneMockHandler({ timezone: "not-set" }),
      http.get("/api/proxy/api/graphs/:graphId/schedules", ({ params }) =>
        HttpResponse.json([
          {
            id: "schedule-id",
            name: "Teig ansetzen",
            user_id: "user-id",
            graph_id: params.graphId,
            graph_version: 1,
            cron: "45 2 * * 1-5",
            timezone: "Asia/Tokyo",
            next_run_time: "2026-11-12T02:45:00+09:00",
            input_data: {},
            input_credentials: {},
          },
        ]),
      ),
    );

    render(<SelectedScheduleView agent={agent} scheduleId="schedule-id" />);

    const browserTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(
      await screen.findByText(getTimezoneDisplayName(browserTimezone)),
    ).toBeDefined();
    expect(screen.queryByText("Not set")).toBeNull();
  });
});
