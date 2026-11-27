import { getGetV1GetUserTimezoneMockHandler } from "@/app/api/__generated__/endpoints/auth/auth.msw";
import {
  getGetV2GetLibraryAgentResponseMock,
  getGetV2ListTriggerAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getTimezoneDisplayName } from "@/lib/timezone-utils";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, test, vi } from "vitest";
import { SelectedScheduleView } from "../components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { SelectedTriggerAgentView } from "../components/NewAgentLibraryView/components/selected-views/SelectedTriggerAgentView/SelectedTriggerAgentView";

describe.each(["schedule", "trigger"] as const)("%s next run", (view) => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

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

  // The cron runs in the schedule's timezone; the next run is shown in the
  // viewer's: their profile's, or the browser's while that is unset.
  test.each([
    {
      viewer: "America/New_York",
      scheduleTimezone: "UTC",
      scheduleLabel: "UTC",
      nextRun: "January 15, 2024 at 07:00",
      nextRunTimezone: "America/New_York",
    },
    {
      viewer: "not-set",
      scheduleTimezone: "America/New_York",
      scheduleLabel: getTimezoneDisplayName("America/New_York"),
      nextRun: "January 15, 2024 at 21:00",
      nextRunTimezone: "Asia/Tokyo",
    },
  ])(
    "labels each time with its own timezone (viewer $viewer, schedule $scheduleTimezone)",
    async ({
      viewer,
      scheduleTimezone,
      scheduleLabel,
      nextRun,
      nextRunTimezone,
    }) => {
      const resolvedOptions = Intl.DateTimeFormat.prototype.resolvedOptions;
      vi.spyOn(
        Intl.DateTimeFormat.prototype,
        "resolvedOptions",
      ).mockImplementation(function (this: Intl.DateTimeFormat) {
        return { ...resolvedOptions.call(this), timeZone: "Asia/Tokyo" };
      });
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
        getGetV1GetUserTimezoneMockHandler({ timezone: viewer }),
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
              timezone: scheduleTimezone,
              next_run_time: "2024-01-15T12:00:00Z",
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

      const nextRunRow = (await screen.findByText("Next run"))
        .parentElement as HTMLElement;
      expect(await within(nextRunRow).findByText(nextRun)).toBeDefined();
      expect(
        within(nextRunRow).getByText(getTimezoneDisplayName(nextRunTimezone)),
      ).toBeDefined();

      const recurrenceRow = screen.getByText("Recurrence")
        .parentElement as HTMLElement;
      expect(within(recurrenceRow).getByText(scheduleLabel)).toBeDefined();

      if (view === "schedule") {
        // The header repeats the recurrence: "<cron> · <schedule timezone>".
        expect(
          screen.getByText((text) => text.endsWith(` · ${scheduleLabel}`)),
        ).toBeDefined();
      }
    },
  );
});

test("the edit dialog says which timezone its time is in", async () => {
  const agent = getGetV2GetLibraryAgentResponseMock({
    id: "agent-id",
    graph_id: "graph-id",
    graph_version: 1,
    name: "Scheduled Agent",
    input_schema: { properties: {} },
    credentials_input_schema: { properties: {} },
  });
  server.use(
    getGetV1GetUserTimezoneMockHandler({ timezone: "Europe/London" }),
    http.get("/api/proxy/api/graphs/:graphId/schedules", ({ params }) =>
      HttpResponse.json([
        {
          id: "schedule-id",
          name: "Morning run",
          user_id: "user-id",
          graph_id: params.graphId,
          graph_version: 1,
          cron: "0 8 * * *",
          timezone: "America/New_York",
          next_run_time: "2024-01-15T13:00:00Z",
          input_data: {},
          input_credentials: {},
        },
      ]),
    ),
  );

  render(<SelectedScheduleView agent={agent} scheduleId="schedule-id" />);

  const [editButton] = await screen.findAllByRole("button", {
    name: /edit schedule/i,
  });
  await userEvent.setup().click(editButton);

  const dialog = await screen.findByRole("dialog");
  expect(within(dialog).getByLabelText("At")).toHaveProperty("value", "08:00");
  expect(
    within(dialog).getByText(
      `Time in ${getTimezoneDisplayName("America/New_York")}`,
    ),
  ).toBeDefined();
});
