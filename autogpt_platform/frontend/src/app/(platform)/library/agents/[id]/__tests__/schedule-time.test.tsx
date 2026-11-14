import { getGetV1GetUserTimezoneMockHandler } from "@/app/api/__generated__/endpoints/auth/auth.msw";
import {
  getGetV2GetLibraryAgentResponseMock,
  getGetV2ListTriggerAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
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
        http.get(
          "/api/proxy/api/internal/graphs/:graphId/schedules",
          ({ params }) =>
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
