import {
  getDeleteWorkspaceFileMockHandler200,
  getListWorkspaceFilesMockHandler200,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { UIMessage } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../../../copilotStreamStore";
import { useCopilotUIStore } from "../../../store";
import { WorkspaceFilesPopover } from "../components/WorkspaceFilesPopover";

// Popover renders through a portal and only shows content while open in
// real usage; mocking it as an always-mounted pair of divs (the pattern
// used by UsagePopover's tests) lets us assert the section content and
// wiring without fighting Radix positioning/portal internals in jsdom.
vi.mock("@/components/molecules/Popover/Popover", () => {
  function Popover({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverTrigger({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverContent({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  return { Popover, PopoverTrigger, PopoverContent };
});

const SESSION = "session-1";

function listResponse(
  files: ListFilesResponse["files"] = [],
): ListFilesResponse {
  return { files, offset: 0, has_more: false };
}

function activityMessages(): UIMessage[] {
  return [
    {
      id: "m1",
      role: "assistant",
      parts: [
        {
          type: "tool-run_agent",
          toolCallId: "call-1",
          state: "output-available",
          input: {},
          output: {
            execution_id: "exec-1",
            status: "RUNNING",
            graph_name: "Daily Digest",
            graph_id: "graph-1",
          },
        },
      ],
    } as unknown as UIMessage,
  ];
}

afterEach(() => {
  useCopilotStreamStore.setState({ messageSnapshots: {} });
});

describe("WorkspaceFilesPopover", () => {
  it("shows the empty state when there are no files, runs, or schedules", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listResponse()));
    render(<WorkspaceFilesPopover sessionId={SESSION} />);

    expect(await screen.findByText("Nothing here yet.")).toBeDefined();
  });

  it("lists files and opens one as an artifact preview on click", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(
        listResponse([
          {
            id: "aaaaaaaa-0000-0000-0000-000000000001",
            name: "result.csv",
            path: "/sessions/session-1/result.csv",
            mime_type: "text/csv",
            size_bytes: 4096,
            metadata: { origin: "agent" },
            origin: "generated",
            created_at: "2026-05-20T11:00:00Z",
          },
        ]),
      ),
    );
    render(<WorkspaceFilesPopover sessionId={SESSION} />);

    fireEvent.click(await screen.findByTitle("result.csv"));

    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact?.title,
    ).toBe("result.csv");
  });

  it("deletes a generated file through the confirm dialog", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(
        listResponse([
          {
            id: "bbbbbbbb-0000-0000-0000-000000000002",
            name: "result.csv",
            path: "/sessions/session-1/result.csv",
            mime_type: "text/csv",
            size_bytes: 4096,
            metadata: { origin: "agent" },
            origin: "generated",
            created_at: "2026-05-20T11:00:00Z",
          },
        ]),
      ),
    );
    let deleted = false;
    server.use(
      getDeleteWorkspaceFileMockHandler200(() => {
        deleted = true;
        return {
          deleted: true,
          file_id: "bbbbbbbb-0000-0000-0000-000000000002",
        };
      }),
    );
    render(<WorkspaceFilesPopover sessionId={SESSION} />);

    fireEvent.click(await screen.findByLabelText("Delete result.csv"));
    fireEvent.click(await screen.findByRole("button", { name: /^Delete$/ }));

    await waitFor(() => expect(deleted).toBe(true));
  });

  it("shows runs and schedules sections alongside the files section", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listResponse()));
    useCopilotStreamStore
      .getState()
      .setMessageSnapshot(SESSION, activityMessages());

    render(<WorkspaceFilesPopover sessionId={SESSION} />);

    expect(await screen.findByText(/^Runs \(1\)/)).toBeDefined();
    expect(screen.getByText("Daily Digest")).toBeDefined();
    expect(screen.queryByText("Nothing here yet.")).toBeNull();
  });
});
