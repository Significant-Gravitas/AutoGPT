import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelToggle } from "../ContextPanelToggle";
import { useSessionFiles } from "../components/FilesTab/useSessionFiles";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const ARTIFACT = {
  id: "f1",
  title: "doc.md",
  mimeType: "text/markdown",
  sourceUrl: "/api/proxy/api/workspace/files/f1/download",
  origin: "agent" as const,
};

function setPanel(overrides: Partial<ReturnType<typeof panelState>>) {
  useCopilotUIStore.setState((s) => ({
    artifactPanel: { ...s.artifactPanel, ...overrides },
  }));
}

function panelState() {
  return useCopilotUIStore.getState().artifactPanel;
}

beforeEach(() => {
  setPanel({
    isOpen: false,
    activeArtifact: null,
    activeTab: "files",
    lastArtifact: null,
    history: [],
  });
});

afterEach(() => {
  setPanel({
    isOpen: false,
    activeArtifact: null,
    activeTab: "files",
    lastArtifact: null,
    history: [],
  });
});

describe("ContextPanelToggle", () => {
  test("renders no workspace-files trigger — the thread chip owns that", () => {
    render(<ContextPanelToggle />);

    expect(screen.queryByLabelText("Open workspace files")).toBeNull();
    expect(screen.queryByLabelText("Workspace files")).toBeNull();
  });

  test("sidebar toggle closes the artifact preview via closeArtifactPanel", () => {
    setPanel({ activeArtifact: ARTIFACT, isOpen: true });
    render(<ContextPanelToggle />);

    fireEvent.click(screen.getByLabelText("Hide artifacts"));

    expect(panelState().activeArtifact).toBeNull();
  });

  test("sidebar toggle restores the last previewed artifact instead of opening the tabs view", () => {
    setPanel({ lastArtifact: ARTIFACT, isOpen: false });
    render(<ContextPanelToggle />);

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().activeArtifact).toEqual(ARTIFACT);
  });

  test("sidebar toggle opens the artifacts tab when there is no remembered artifact", () => {
    setPanel({ lastArtifact: null, isOpen: false });
    render(<ContextPanelToggle />);

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeTab).toBe("artifacts");
  });

  test("sidebar toggle reads as active when the artifacts tab is open", () => {
    setPanel({ isOpen: true, activeTab: "artifacts" });
    render(<ContextPanelToggle />);

    expect(screen.getByLabelText("Hide artifacts")).toBeDefined();
  });
});

const SESSION = "session-1";

// The toggle renders the same bare icon whether the files are still in
// flight or every one of them was filtered out, so a "stays unlabeled"
// assertion would pass trivially. This probe gives those tests something to
// await that proves the listing actually arrived.
function LoadedFilesProbe({ sessionId }: { sessionId: string }) {
  const { generated } = useSessionFiles(sessionId);
  return <div data-testid="generated-count">{generated.length}</div>;
}

function realFile(): ListFilesResponse["files"][number] {
  return {
    id: "aaaaaaaa-0000-0000-0000-000000000001",
    name: "result.csv",
    path: "/sessions/session-1/result.csv",
    mime_type: "text/csv",
    size_bytes: 4096,
    metadata: { origin: "agent" },
    origin: "generated",
    created_at: "2026-05-20T11:00:00Z",
  };
}

// One entry per branch of `isInternalToolOutput`, each newer than the real
// deliverable so a missing filter would surface it instead.
function toolOutputs(): ListFilesResponse["files"] {
  return [
    {
      id: "bbbbbbbb-0000-0000-0000-000000000002",
      name: "toolu_01ABCdef.json",
      path: "/sessions/session-1/tool-outputs/toolu_01ABCdef.json",
      mime_type: "application/json",
      size_bytes: 512,
      metadata: {},
      origin: "generated",
      created_at: "2026-05-20T12:00:00Z",
    },
    {
      id: "cccccccc-0000-0000-0000-000000000003",
      name: "mcp_a1b2-c3d4.json",
      path: "/sessions/session-1/tool-outputs/mcp_a1b2-c3d4.json",
      mime_type: "application/json",
      size_bytes: 256,
      metadata: {},
      origin: "generated",
      created_at: "2026-05-20T13:00:00Z",
    },
    {
      id: "dddddddd-0000-0000-0000-000000000004",
      name: "toolu_02ZYXwvu.json",
      path: "/sessions/session-1/tool-results/toolu_02ZYXwvu.json",
      mime_type: "application/json",
      size_bytes: 128,
      metadata: {},
      origin: "generated",
      created_at: "2026-05-20T14:00:00Z",
    },
  ];
}

function listing(files: ListFilesResponse["files"]): ListFilesResponse {
  return { files, offset: 0, has_more: false };
}

describe("ContextPanelToggle internal tool output", () => {
  test("wears the newest user-facing file, not a newer tool output", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(
        listing([realFile(), ...toolOutputs()]),
      ),
    );
    render(<ContextPanelToggle sessionId={SESSION} />);

    const button = await screen.findByLabelText("Open result.csv");
    fireEvent.click(button);

    expect(panelState().activeArtifact?.id).toBe(
      "aaaaaaaa-0000-0000-0000-000000000001",
    );
  });

  test("still wears a deliverable whose name looks like an SDK tool id", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(
        listing([
          realFile(),
          ...toolOutputs(),
          {
            id: "eeeeeeee-0000-0000-0000-000000000005",
            name: "mcp_config.json",
            path: "/sessions/session-1/mcp_config.json",
            mime_type: "application/json",
            size_bytes: 64,
            metadata: {},
            origin: "generated",
            created_at: "2026-05-20T15:00:00Z",
          },
        ]),
      ),
    );
    render(<ContextPanelToggle sessionId={SESSION} />);

    expect(await screen.findByLabelText("Open mcp_config.json")).toBeDefined();
  });

  test("stays unlabeled when every generated file is an internal tool output", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listing(toolOutputs())));
    render(
      <>
        <ContextPanelToggle sessionId={SESSION} />
        <LoadedFilesProbe sessionId={SESSION} />
      </>,
    );

    await waitFor(() =>
      expect(screen.getByTestId("generated-count").textContent).toBe("3"),
    );
    expect(screen.getByLabelText("Open artifacts")).toBeDefined();

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().activeArtifact).toBeNull();
    expect(panelState().activeTab).toBe("artifacts");
  });
});
