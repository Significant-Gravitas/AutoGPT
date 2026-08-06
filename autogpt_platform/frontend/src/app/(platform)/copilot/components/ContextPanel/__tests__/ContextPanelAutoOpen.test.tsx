import { beforeEach, describe, expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelAutoOpen } from "../ContextPanelAutoOpen";
import { useSessionFiles } from "../components/FilesTab/useSessionFiles";

const SESSION = "session-1";

// The auto-open hook renders nothing, so a "the panel stays closed" assertion
// would pass trivially before the file request resolves. This probe gives the
// test something to await that proves the files actually arrived.
function LoadedFilesProbe({ sessionId }: { sessionId: string }) {
  const { generated } = useSessionFiles(sessionId);
  return <div data-testid="generated-count">{generated.length}</div>;
}

function withFiles(): ListFilesResponse {
  return {
    files: [
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
    ],
    offset: 0,
    has_more: false,
  };
}

// One entry per branch of `isInternalToolOutput`, each newer than the real
// deliverable so a missing filter would surface it instead.
function withToolOutputAndRealFile(): ListFilesResponse {
  const response = withFiles();
  response.files.push(
    {
      id: "bbbbbbbb-0000-0000-0000-000000000002",
      name: "toolu_01ABCdef.json",
      path: "/sessions/session-1/toolu_01ABCdef.json",
      mime_type: "application/json",
      size_bytes: 512,
      metadata: {},
      origin: "generated",
      created_at: "2026-05-20T12:00:00Z",
    },
    {
      id: "cccccccc-0000-0000-0000-000000000003",
      name: "mcp_a1b2-c3d4.json",
      path: "/sessions/session-1/mcp_a1b2-c3d4.json",
      mime_type: "application/json",
      size_bytes: 256,
      metadata: {},
      origin: "generated",
      created_at: "2026-05-20T13:00:00Z",
    },
    {
      id: "dddddddd-0000-0000-0000-000000000004",
      name: "summary.json",
      path: "/root/.claude/projects/-workspace/abc-123/tool-results/summary.json",
      mime_type: "application/json",
      size_bytes: 128,
      metadata: {},
      origin: "generated",
      created_at: "2026-05-20T14:00:00Z",
    },
  );
  return response;
}

function withOnlyToolOutput(): ListFilesResponse {
  const response = withToolOutputAndRealFile();
  response.files = response.files.filter((file) => file.name !== "result.csv");
  return response;
}

describe("ContextPanelAutoOpen", () => {
  beforeEach(() => {
    useCopilotUIStore.getState().resetAutoOpenState?.();
    useCopilotUIStore.setState((s) => ({
      artifactPanel: {
        ...s.artifactPanel,
        isOpen: false,
        activeArtifact: null,
      },
    }));
  });

  test("opens the last generated file in the artifact panel when the session has files", async () => {
    server.use(getListWorkspaceFilesMockHandler200(withFiles()));
    render(<ContextPanelAutoOpen sessionId={SESSION} />);
    await waitFor(() =>
      expect(
        useCopilotUIStore.getState().artifactPanel.activeArtifact?.id,
      ).toBe("aaaaaaaa-0000-0000-0000-000000000001"),
    );
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });

  test("skips newer internal tool outputs and opens the real generated file", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(withToolOutputAndRealFile()),
    );
    render(<ContextPanelAutoOpen sessionId={SESSION} />);
    await waitFor(() =>
      expect(
        useCopilotUIStore.getState().artifactPanel.activeArtifact?.id,
      ).toBe("aaaaaaaa-0000-0000-0000-000000000001"),
    );
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });

  test("leaves the panel closed when every generated file is an internal tool output", async () => {
    server.use(getListWorkspaceFilesMockHandler200(withOnlyToolOutput()));
    render(
      <>
        <ContextPanelAutoOpen sessionId={SESSION} />
        <LoadedFilesProbe sessionId={SESSION} />
      </>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("generated-count").textContent).toBe("3"),
    );
    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact,
    ).toBeNull();
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });
});
