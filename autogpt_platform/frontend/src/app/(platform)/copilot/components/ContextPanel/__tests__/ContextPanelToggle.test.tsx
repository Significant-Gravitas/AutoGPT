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
import { Key, storage } from "@/services/storage/local-storage";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelToggle } from "../ContextPanelToggle";
import { useSessionFiles } from "../components/FilesTab/useSessionFiles";
import { useWorkspaceFileCards } from "../../WorkspaceFileCards/useWorkspaceFileCards";
import { WorkspaceFileCard } from "../../WorkspaceFileCards/components/WorkspaceFileCard";
import { ArtifactsTab } from "../components/ArtifactsTab/ArtifactsTab";

let mobile = false;
function setMobile(value: boolean) {
  mobile = value;
}
vi.mock("../../../useIsMobile", () => ({
  useIsMobile: () => mobile,
}));

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
  server.use(
    getListWorkspaceFilesMockHandler200({
      files: [],
      offset: 0,
      has_more: false,
    }),
  );
  setPanel({
    isOpen: false,
    activeArtifact: null,
    activeTab: "files",
    lastArtifact: null,
    history: [],
    mode: "artifact",
    isComputerOpen: false,
    computer: null,
  });
});

afterEach(() => {
  setMobile(false);
  setPanel({
    isOpen: false,
    activeArtifact: null,
    activeTab: "files",
    lastArtifact: null,
    history: [],
    mode: "artifact",
    isComputerOpen: false,
    computer: null,
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

  test("sidebar toggle turns the computer face back to the artifact under it", () => {
    setPanel({
      activeArtifact: ARTIFACT,
      isOpen: true,
      mode: "computer",
      isComputerOpen: true,
    });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().mode).toBe("artifact");
    expect(panelState().isComputerOpen).toBe(false);
    expect(panelState().activeArtifact).toEqual(ARTIFACT);
    expect(panelState().isOpen).toBe(true);
  });

  test("sidebar toggle turns the computer face back to the remembered preview", () => {
    setPanel({
      lastArtifact: ARTIFACT,
      isOpen: true,
      activeTab: "files",
      mode: "computer",
      isComputerOpen: true,
    });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().activeArtifact).toEqual(ARTIFACT);
    expect(panelState().isComputerOpen).toBe(false);
    expect(panelState().mode).toBe("artifact");
  });

  test("a panel the computer opened is stored as open once turned to the library", () => {
    storage.set(Key.COPILOT_CONTEXT_PANEL_OPEN, "false");
    setPanel({ isOpen: false, activeTab: "artifacts" });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open computer"));
    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().isOpen).toBe(true);
    expect(storage.get(Key.COPILOT_CONTEXT_PANEL_OPEN)).toBe("true");
  });

  test("sidebar toggle turns the computer face to the library when nothing is under it", () => {
    setPanel({
      isOpen: true,
      activeTab: "artifacts",
      mode: "computer",
      isComputerOpen: true,
    });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().isComputerOpen).toBe(false);
    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeTab).toBe("artifacts");
  });
});

describe("ContextPanelToggle computer button", () => {
  test("is there for a chat with a session and absent without one", () => {
    const { unmount } = render(<ContextPanelToggle />);
    expect(screen.queryByLabelText("Open computer")).toBeNull();
    unmount();

    render(<ContextPanelToggle sessionId="s1" />);
    expect(screen.getByLabelText("Open computer")).toBeDefined();
  });

  test("opens the computer face with no artifact and no desktop started", () => {
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open computer"));

    expect(panelState().isOpen).toBe(true);
    expect(panelState().mode).toBe("computer");
    expect(panelState().isComputerOpen).toBe(true);
  });

  test("opens the computer face over an artifact without dropping it", () => {
    setPanel({ activeArtifact: ARTIFACT, isOpen: true });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open computer"));

    expect(panelState().mode).toBe("computer");
    expect(panelState().isComputerOpen).toBe(true);
    expect(panelState().activeArtifact).toEqual(ARTIFACT);
    expect(screen.getByLabelText("Hide computer")).toBeDefined();
    expect(screen.getByLabelText("Open artifacts")).toBeDefined();
  });

  test("hiding the computer closes the panel it opened over, keeping the remembered preview", () => {
    setPanel({ isOpen: false, lastArtifact: ARTIFACT });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open computer"));
    fireEvent.click(screen.getByLabelText("Hide computer"));

    expect(panelState().isComputerOpen).toBe(false);
    expect(panelState().isOpen).toBe(false);
    expect(panelState().lastArtifact).toEqual(ARTIFACT);
  });

  test("hiding the computer returns to the tab that was open under it", () => {
    setPanel({ isOpen: true, activeTab: "artifacts" });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open computer"));
    fireEvent.click(screen.getByLabelText("Hide computer"));

    expect(panelState().isComputerOpen).toBe(false);
    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeTab).toBe("artifacts");
    expect(screen.getByLabelText("Hide artifacts")).toBeDefined();
  });

  test("hiding the computer reveals the preview it was covering, history intact", () => {
    const earlier = { ...ARTIFACT, id: "f0", title: "earlier.md" };
    setPanel({
      activeArtifact: ARTIFACT,
      history: [earlier],
      isOpen: true,
      mode: "computer",
      isComputerOpen: true,
    });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Hide computer"));

    expect(panelState().isComputerOpen).toBe(false);
    expect(panelState().mode).toBe("artifact");
    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeArtifact).toEqual(ARTIFACT);
    expect(panelState().history).toEqual([earlier]);
  });

  test("stays while the workspace files card is open, alone", () => {
    setPanel({ isOpen: true, activeTab: "files" });
    render(<ContextPanelToggle sessionId="s1" />);

    expect(screen.queryByLabelText("Open artifacts")).toBeNull();
    fireEvent.click(screen.getByLabelText("Open computer"));

    expect(panelState().isComputerOpen).toBe(true);
    expect(panelState().mode).toBe("computer");
  });

  test("is hidden on mobile, whose sheet has no computer face", () => {
    setMobile(true);
    render(<ContextPanelToggle sessionId="s1" />);

    expect(screen.queryByLabelText("Open computer")).toBeNull();
    expect(screen.getByLabelText("Open artifacts")).toBeDefined();
  });
});

const SESSION = "session-1";

// The toggle renders the same bare icon whether the files are still in
// flight or every one of them was filtered out, so a "stays unlabeled"
// assertion would pass trivially. This probe gives those tests something to
// await that proves the listing actually arrived.
interface Props {
  sessionId: string;
}

function LoadedFilesProbe({ sessionId }: Props) {
  const { generated } = useSessionFiles(sessionId);
  return <div data-testid="generated-count">{generated.length}</div>;
}

function WorkspaceFilesProbe({ sessionId }: Props) {
  const { files, handleOpen, handleDownload, setPendingDelete } =
    useWorkspaceFileCards(sessionId);
  return (
    <>
      {files.map((file) => (
        <WorkspaceFileCard
          key={file.item.id}
          file={file}
          onOpen={handleOpen}
          onDownload={handleDownload}
          onRequestDelete={setPendingDelete}
        />
      ))}
    </>
  );
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
    // The id an SDK-transport session actually produces: `_execute_tool_sync`
    // synthesizes `sdk-<uuid>` rather than forwarding the SDK's `toolu_*`, so
    // this is the shape the default AutoPilot path parks in the directory.
    {
      id: "ffffffff-0000-0000-0000-000000000006",
      name: "sdk-a1b2c3d4e5f6.json",
      path: "/sessions/session-1/tool-outputs/sdk-a1b2c3d4e5f6.json",
      mime_type: "application/json",
      size_bytes: 320,
      metadata: {},
      origin: "generated",
      created_at: "2026-05-20T16:00:00Z",
    },
  ];
}

function listing(files: ListFilesResponse["files"]): ListFilesResponse {
  return { files, offset: 0, has_more: false };
}

describe("ContextPanelToggle internal tool output", () => {
  test("uses provenance after an internal output is renamed, while retaining file access", async () => {
    const internal = {
      ...toolOutputs()[0],
      name: "renamed.json",
      path: "/sessions/session-1/renamed.json",
      metadata: { purpose: "tool-output" },
    };
    server.use(
      getListWorkspaceFilesMockHandler200(listing([realFile(), internal])),
    );
    render(
      <>
        <ContextPanelToggle sessionId={SESSION} />
        <ArtifactsTab sessionId={SESSION} />
        <WorkspaceFilesProbe sessionId={SESSION} />
      </>,
    );

    expect(await screen.findByLabelText("Open result.csv")).toBeDefined();
    expect(await screen.findByLabelText("Download renamed.json")).toBeDefined();
    expect(screen.getAllByText("renamed.json")).toHaveLength(1);
    fireEvent.click(screen.getByTitle("renamed.json"));
    expect(panelState().activeArtifact?.id).toBe(internal.id);
  });

  test("promotes an explicitly marked deliverable in the legacy tool-output directory", async () => {
    const deliverable = {
      ...toolOutputs()[0],
      metadata: { purpose: "deliverable" },
    };
    server.use(
      getListWorkspaceFilesMockHandler200(listing([realFile(), deliverable])),
    );
    render(<ContextPanelToggle sessionId={SESSION} />);

    expect(
      await screen.findByLabelText(`Open ${deliverable.name}`),
    ).toBeDefined();
  });
  test("wears the newest user-facing file, not a newer tool output", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(
        listing([realFile(), ...toolOutputs()]),
      ),
    );
    render(<ContextPanelToggle sessionId={SESSION} />);

    const button = await screen.findByLabelText("Open result.csv");
    fireEvent.click(button);

    expect(panelState().isOpen).toBe(true);
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
            created_at: "2026-05-20T17:00:00Z",
          },
        ]),
      ),
    );
    render(<ContextPanelToggle sessionId={SESSION} />);

    expect(await screen.findByLabelText("Open mcp_config.json")).toBeDefined();
  });

  test("still wears a deliverable under the user's own tool-outputs folder", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(
        listing([
          realFile(),
          ...toolOutputs(),
          {
            id: "99999999-0000-0000-0000-000000000007",
            name: "data.json",
            path: "/sessions/session-1/my-pipeline/tool-outputs/data.json",
            mime_type: "application/json",
            size_bytes: 96,
            metadata: {},
            origin: "generated",
            created_at: "2026-05-20T17:00:00Z",
          },
        ]),
      ),
    );
    render(<ContextPanelToggle sessionId={SESSION} />);

    expect(await screen.findByLabelText("Open data.json")).toBeDefined();
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
      expect(screen.getByTestId("generated-count").textContent).toBe(
        String(toolOutputs().length),
      ),
    );
    expect(screen.getByLabelText("Open artifacts")).toBeDefined();

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().activeArtifact).toBeNull();
    expect(panelState().activeTab).toBe("artifacts");
  });
});
