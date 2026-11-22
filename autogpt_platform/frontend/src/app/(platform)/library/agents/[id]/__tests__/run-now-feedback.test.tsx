import { getGetV1GetUserTimezoneMockHandler } from "@/app/api/__generated__/endpoints/auth/auth.msw";
import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, test } from "vitest";
import { SelectedRunActions } from "../components/NewAgentLibraryView/components/selected-views/SelectedRunView/components/SelectedRunActions/SelectedRunActions";
import { SelectedScheduleView } from "../components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/SelectedScheduleView";
import { ScheduleActionsDropdown } from "../components/NewAgentLibraryView/components/sidebar/SidebarRunsList/components/ScheduleActionsDropdown";

// Run now (from a schedule's view or its row in the sidebar) and Rerun task
// start a run with what was saved: the inputs and the key. The server can
// refuse it, e.g. when that key was removed, so none of them may say the run
// started until it has.

const REMOVED_KEY = {
  id: "removed-key",
  provider: "unreal_speech",
  type: "api_key",
  title: "Narration key",
} as const;

const agent = getGetV2GetLibraryAgentResponseMock({
  id: "agent-id",
  graph_id: "graph-id",
  graph_version: 1,
  name: "Morning narration",
  input_schema: { properties: {} },
  credentials_input_schema: { properties: {} },
  trigger_setup_info: null,
});

const schedule = {
  id: "schedule-id",
  name: "Every morning",
  user_id: "user-id",
  graph_id: "graph-id",
  graph_version: 1,
  cron: "0 8 * * *",
  timezone: "UTC",
  next_run_time: "2030-01-15T08:00:00Z",
  input_data: {},
  input_credentials: { credentials: REMOVED_KEY },
} as unknown as GraphExecutionJobInfo;

const pastRun = {
  id: "past-run-id",
  graph_id: "graph-id",
  graph_version: 1,
  status: "COMPLETED",
  inputs: {},
  credential_inputs: { credentials: REMOVED_KEY },
} as unknown as GraphExecution;

const surfaces = [
  {
    name: "Run now in a schedule's view",
    show() {
      server.use(
        getGetV1GetUserTimezoneMockHandler({ timezone: "UTC" }),
        http.get("/api/proxy/api/graphs/:graphId/schedules", () =>
          HttpResponse.json([schedule]),
        ),
      );
      render(
        <>
          <SelectedScheduleView agent={agent} scheduleId={schedule.id} />
          <Toaster />
        </>,
      );
    },
    async start() {
      fireEvent.click(await screen.findByRole("button", { name: "Run now" }));
    },
  },
  {
    name: "Run now on a schedule's row",
    show() {
      render(
        <>
          <ScheduleActionsDropdown agent={agent} schedule={schedule} />
          <Toaster />
        </>,
      );
    },
    async start() {
      // Radix DropdownMenu opens on pointerdown, not click, under happy-dom.
      fireEvent.pointerDown(screen.getByLabelText("More actions"), {
        button: 0,
      });
      fireEvent.click(await screen.findByText("Run now"));
    },
  },
  {
    name: "Rerun task in a run's view",
    show() {
      render(
        <>
          <SelectedRunActions agent={agent} run={pastRun} />
          <Toaster />
        </>,
      );
    },
    async start() {
      fireEvent.click(
        await screen.findByRole("button", { name: "Rerun task" }),
      );
    },
  },
];

// Answers the run request only when the test says so, so it can look at
// the page while the request is still on its way.
function holdRunRequest(answer: () => Response) {
  let release!: () => void;
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  let received = false;
  let handled!: () => void;
  const answered = new Promise<void>((resolve) => {
    handled = resolve;
  });
  server.use(
    http.post("/api/proxy/api/graphs/:graphId/execute/:version", async () => {
      received = true;
      await released;
      handled();
      return answer();
    }),
  );
  return {
    release,
    wasReceived: () => received,
    /** Lets a held request finish, so none is left open when a test ends,
     * whether or not its assertions passed. */
    async settle() {
      release();
      if (received) await answered;
    },
  };
}

function refusedForTheRemovedKey() {
  return HttpResponse.json(
    {
      detail: {
        type: "validation_error",
        message: "Workflow validation failed: 1 issues on 1 nodes",
        node_errors: {
          "speech-node": {
            credentials: `Unknown credentials #${REMOVED_KEY.id}`,
          },
        },
      },
    },
    { status: 400 },
  );
}

function started() {
  return HttpResponse.json({
    id: "new-run-id",
    graph_id: "graph-id",
    graph_version: 1,
    status: "QUEUED",
  });
}

describe.each(surfaces)("$name", (surface) => {
  test("says nothing started while the run is being started, nor once the server refuses it", async () => {
    const run = holdRunRequest(refusedForTheRemovedKey);
    try {
      surface.show();

      await surface.start();
      await waitFor(() => expect(run.wasReceived()).toBe(true));
      expect(
        screen.queryByText("Run started"),
        "nothing has started while the request is pending",
      ).toBeNull();

      run.release();
      expect(await screen.findByText("Failed to start run")).toBeDefined();
      expect(screen.queryByText("Run started")).toBeNull();
    } finally {
      await run.settle();
    }
  });

  test("says the run started once the server has started it", async () => {
    const run = holdRunRequest(started);
    try {
      surface.show();

      await surface.start();
      await waitFor(() => expect(run.wasReceived()).toBe(true));
      expect(screen.queryByText("Run started")).toBeNull();

      run.release();
      expect(await screen.findByText("Run started")).toBeDefined();
      expect(screen.queryByText("Failed to start run")).toBeNull();
    } finally {
      await run.settle();
    }
  });
});
