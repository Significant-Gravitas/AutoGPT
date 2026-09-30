import { getPostV1ExecuteGraphAgentMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, test, vi } from "vitest";
import { ScheduleActionsDropdown } from "./ScheduleActionsDropdown";

const sendDatafastEvent = vi.hoisted(() => vi.fn());
vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent },
}));

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    useToast: () => ({ toast: toastMock }),
  };
});

const agent = {
  id: "lib-1",
  graph_id: "graph-xyz",
  graph_version: 1,
  name: "My agent",
} as unknown as LibraryAgent;

const schedule = {
  id: "sched-1",
  graph_id: "graph-xyz",
  graph_version: 1,
  input_data: {},
  input_credentials: {},
} as unknown as GraphExecutionJobInfo;

afterEach(() => {
  toastMock.mockClear();
  sendDatafastEvent.mockClear();
  server.resetHandlers();
});

describe("ScheduleActionsDropdown", () => {
  test("Run now records a run_agent goal from the library surface", async () => {
    server.use(
      getPostV1ExecuteGraphAgentMockHandler({
        id: "run-1",
        graph_id: "graph-xyz",
      } as GraphExecutionMeta),
    );
    const onRunCreated = vi.fn();

    render(
      <ScheduleActionsDropdown
        agent={agent}
        schedule={schedule}
        onRunCreated={onRunCreated}
      />,
    );

    // Radix DropdownMenu opens on pointerdown, not click, under happy-dom.
    fireEvent.pointerDown(screen.getByLabelText("More actions"), {
      button: 0,
    });
    fireEvent.click(await screen.findByText("Run now"));

    await waitFor(() => {
      expect(onRunCreated).toHaveBeenCalledWith("run-1");
    });
    expect(sendDatafastEvent).toHaveBeenCalledExactlyOnceWith("run_agent", {
      id: "graph-xyz",
      name: "My agent",
      surface: "library",
    });
  });
});
