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
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../../../copilotStreamStore";
import { useCopilotUIStore } from "../../../store";
import { WorkspaceFileCards } from "../WorkspaceFileCards";

// Skip real download wiring — fetch/blob plumbing isn't under test.
const downloadArtifactMock = vi.fn(() => Promise.resolve());
vi.mock("../../ArtifactPanel/downloadArtifact", () => ({
  downloadArtifact: (...args: unknown[]) =>
    downloadArtifactMock(...(args as [])),
}));

const downloadFilesAsZipMock = vi.fn(() => Promise.resolve());
vi.mock(
  "../../ContextPanel/components/FilesTab/helpers",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("../../ContextPanel/components/FilesTab/helpers")
      >();
    return {
      ...actual,
      downloadFilesAsZip: (...args: unknown[]) =>
        downloadFilesAsZipMock(...(args as [])),
    };
  },
);

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastSpy(...(args as [])),
  useToast: () => ({ toast: toastSpy }),
}));

const SESSION = "session-1";

function listResponse(): ListFilesResponse {
  return {
    files: [
      {
        id: "aaaaaaaa-0000-0000-0000-000000000001",
        name: "uploaded.png",
        path: "/sessions/session-1/uploaded.png",
        mime_type: "image/png",
        size_bytes: 1024,
        metadata: { origin: "user-upload" },
        origin: "uploaded",
        created_at: "2026-05-20T10:00:00Z",
      },
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
    ],
    offset: 0,
    has_more: false,
  };
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
        {
          type: "tool-schedule_followup",
          toolCallId: "call-2",
          state: "output-available",
          input: { message: "Check the inbox", cron: "0 9 * * *" },
          output: {
            type: "schedule_created",
            schedule_id: "sched-1",
            name: "Morning brief",
            next_run_time: "2026-05-21T09:00:00Z",
            cron: "0 9 * * *",
          },
        },
      ],
    } as unknown as UIMessage,
  ];
}

beforeEach(() => {
  server.use(getListWorkspaceFilesMockHandler200(listResponse()));
});

afterEach(() => {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      activeArtifact: null,
      history: [],
      activeTab: "files",
      lastArtifact: null,
      mode: "artifact",
      computer: null,
      isComputerOpen: false,
    },
  });
  useCopilotStreamStore.setState({ messageSnapshots: {} });
  downloadArtifactMock.mockClear();
  downloadArtifactMock.mockImplementation(() => Promise.resolve());
  downloadFilesAsZipMock.mockClear();
  downloadFilesAsZipMock.mockImplementation(() => Promise.resolve());
  toastSpy.mockClear();
});

describe("WorkspaceFileCards", () => {
  it("renders the session's files", async () => {
    render(<WorkspaceFileCards sessionId={SESSION} />);

    expect(await screen.findByText("uploaded.png")).toBeDefined();
    expect(screen.getByText("result.csv")).toBeDefined();
    expect(screen.getByText(/^Files in this chat \(2\)/)).toBeDefined();
    expect(screen.getByLabelText("Download all")).toBeDefined();
  });

  it("lists generated documents from all of the expert's chats", async () => {
    const expertRequests: URLSearchParams[] = [];
    server.use(
      http.get("*/api/workspace/files", ({ request }) => {
        const params = new URL(request.url).searchParams;
        if (params.get("expert_id") !== "expert-maria") {
          return HttpResponse.json(listResponse());
        }
        expertRequests.push(params);
        return HttpResponse.json({
          files: [
            ...Array.from({ length: 5 }, (_, index) => ({
              id: `expert-document-${index}`,
              name: `maria-${index + 1}.md`,
              path: `/sessions/maria-${index}/document.md`,
              mime_type: "text/markdown",
              size_bytes: 128,
              origin: "generated" as const,
              created_at: "2026-09-29T10:00:00Z",
              expert_id: "expert-maria",
            })),
            {
              id: "expert-tool-output",
              name: "raw.json",
              path: "/sessions/maria/tool-results/raw.json",
              mime_type: "application/json",
              size_bytes: 64,
              metadata: { purpose: "tool-output" },
              origin: "generated" as const,
              created_at: "2026-09-29T09:00:00Z",
              expert_id: "expert-maria",
            },
          ],
          offset: 0,
          has_more: false,
        });
      }),
    );

    render(
      <WorkspaceFileCards
        sessionId={SESSION}
        expert={{ id: "expert-maria", name: "Maria" }}
      />,
    );

    expect(await screen.findByText("All Maria's documents")).toBeDefined();
    expect(await screen.findByText("maria-1.md")).toBeDefined();
    expect(screen.getByText("View more (1)")).toBeDefined();
    expect(screen.queryByText("raw.json")).toBeNull();
    expect(expertRequests).toHaveLength(1);
    expect(expertRequests[0].get("origin")).toBe("generated");
    expect(expertRequests[0].get("limit")).toBe("200");
  });

  it("opens a file as an artifact preview on click", async () => {
    render(<WorkspaceFileCards sessionId={SESSION} />);

    fireEvent.click(await screen.findByTitle("result.csv"));

    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact?.title,
    ).toBe("result.csv");
  });

  it("deletes a generated file through the confirm dialog", async () => {
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
    render(<WorkspaceFileCards sessionId={SESSION} />);

    fireEvent.click(await screen.findByLabelText("Delete result.csv"));
    fireEvent.click(await screen.findByRole("button", { name: /^Delete$/ }));

    await waitFor(() => expect(deleted).toBe(true));
  });

  it("offers no delete for uploaded files", async () => {
    render(<WorkspaceFileCards sessionId={SESSION} />);

    await screen.findByText("uploaded.png");
    expect(screen.queryByLabelText("Delete uploaded.png")).toBeNull();
  });

  it("shows the runs and schedules this chat set in motion", async () => {
    useCopilotStreamStore
      .getState()
      .setMessageSnapshot(SESSION, activityMessages());
    render(<WorkspaceFileCards sessionId={SESSION} />);

    expect(await screen.findByText(/^Runs \(1\)/)).toBeDefined();
    expect(screen.getByText("Daily Digest")).toBeDefined();
    expect(screen.getByText(/^Schedules \(1\)/)).toBeDefined();
    expect(screen.getByText("Morning brief")).toBeDefined();
  });

  it("downloads a single file on click", async () => {
    render(<WorkspaceFileCards sessionId={SESSION} />);

    fireEvent.click(await screen.findByLabelText("Download result.csv"));

    await waitFor(() => expect(downloadArtifactMock).toHaveBeenCalledTimes(1));
    expect(toastSpy).not.toHaveBeenCalled();
  });

  it("toasts when a single-file download fails", async () => {
    downloadArtifactMock.mockImplementation(() =>
      Promise.reject(new Error("network error")),
    );
    render(<WorkspaceFileCards sessionId={SESSION} />);

    fireEvent.click(await screen.findByLabelText("Download uploaded.png"));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Download failed" }),
      ),
    );
  });

  it("zips and downloads every file via Download all", async () => {
    render(<WorkspaceFileCards sessionId={SESSION} />);

    fireEvent.click(await screen.findByLabelText("Download all"));

    await waitFor(() =>
      expect(downloadFilesAsZipMock).toHaveBeenCalledTimes(1),
    );
    expect(toastSpy).not.toHaveBeenCalled();
  });

  it("toasts when Download all fails", async () => {
    downloadFilesAsZipMock.mockImplementation(() =>
      Promise.reject(new Error("zip error")),
    );
    render(<WorkspaceFileCards sessionId={SESSION} />);

    fireEvent.click(await screen.findByLabelText("Download all"));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Download all failed" }),
      ),
    );
  });

  it("toasts and still closes the dialog when delete fails", async () => {
    server.use(
      http.delete("/api/proxy/api/workspace/files/:fileId", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    render(<WorkspaceFileCards sessionId={SESSION} />);

    fireEvent.click(await screen.findByLabelText("Delete result.csv"));
    fireEvent.click(await screen.findByRole("button", { name: /^Delete$/ }));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Failed to delete file" }),
      ),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});
