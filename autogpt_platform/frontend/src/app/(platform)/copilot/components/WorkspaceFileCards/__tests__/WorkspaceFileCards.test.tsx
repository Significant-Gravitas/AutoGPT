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
import { act } from "@testing-library/react";
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

function openFilesCard() {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: true,
      activeArtifact: null,
      history: [],
      activeTab: "files",
      lastArtifact: null,
    },
  });
}

function closeFilesCard() {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      activeArtifact: null,
      history: [],
      activeTab: "files",
      lastArtifact: null,
    },
  });
}

// Long enough for a mounted list query to reach the MSW handler.
function flushRequests() {
  return new Promise((resolve) => setTimeout(resolve, 20));
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
  openFilesCard();
});

afterEach(() => {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      activeArtifact: null,
      history: [],
      activeTab: "files",
      lastArtifact: null,
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
  it("renders the session's files as a floating card when the store flag is open", async () => {
    render(<WorkspaceFileCards sessionId={SESSION} />);

    expect(await screen.findByText("uploaded.png")).toBeDefined();
    expect(screen.getByText("result.csv")).toBeDefined();
    expect(screen.getByText(/^Files \(2\)/)).toBeDefined();
    expect(screen.getByLabelText("Download all")).toBeDefined();
  });

  it("renders nothing while the card is closed", () => {
    useCopilotUIStore.setState({
      artifactPanel: {
        isOpen: false,
        activeArtifact: null,
        history: [],
        activeTab: "files",
        lastArtifact: null,
      },
    });
    const { container } = render(<WorkspaceFileCards sessionId={SESSION} />);
    expect(container.textContent).toBe("");
  });

  it("steps aside while the artifacts tab owns the side panel", () => {
    useCopilotUIStore.setState({
      artifactPanel: {
        isOpen: true,
        activeArtifact: null,
        history: [],
        activeTab: "artifacts",
        lastArtifact: null,
      },
    });
    const { container } = render(<WorkspaceFileCards sessionId={SESSION} />);
    expect(container.textContent).toBe("");
  });

  it("requests the file list only while the card is open", async () => {
    let listRequests = 0;
    server.use(
      getListWorkspaceFilesMockHandler200(() => {
        listRequests += 1;
        return listResponse();
      }),
    );
    closeFilesCard();
    render(<WorkspaceFileCards sessionId={SESSION} />);
    await flushRequests();
    expect(listRequests).toBe(0);

    act(() => openFilesCard());

    expect(await screen.findByText("uploaded.png")).toBeDefined();
    expect(listRequests).toBeGreaterThan(0);
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
