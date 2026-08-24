import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelToggle } from "../ContextPanelToggle";

const flagState = vi.hoisted(() => ({ newToolUI: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.NEW_TOOL_UI ? flagState.newToolUI : false,
  };
});

const SESSION = "session-1";

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
  });
});

afterEach(() => {
  flagState.newToolUI = false;
  setPanel({
    isOpen: false,
    activeArtifact: null,
    activeTab: "files",
    lastArtifact: null,
    history: [],
  });
});

describe("ContextPanelToggle - legacy (flag off)", () => {
  test("renders an open-panel button when closed with no artifact", () => {
    render(<ContextPanelToggle sessionId={SESSION} />);

    const button = screen.getByLabelText("Open workspace panel");
    fireEvent.click(button);

    expect(panelState().isOpen).toBe(true);
  });

  test("renders nothing while the panel is open", () => {
    setPanel({ isOpen: true });
    const { container } = render(<ContextPanelToggle sessionId={SESSION} />);
    expect(container.textContent).toBe("");
  });

  test("renders nothing while an artifact is previewing", () => {
    setPanel({ activeArtifact: ARTIFACT });
    const { container } = render(<ContextPanelToggle sessionId={SESSION} />);
    expect(container.textContent).toBe("");
  });
});

describe("ContextPanelToggle - new tool UI (flag on)", () => {
  beforeEach(() => {
    flagState.newToolUI = true;
  });

  test("shows a files toggle button when the right sidebar is closed", () => {
    render(<ContextPanelToggle sessionId={SESSION} />);

    const filesButton = screen.getByLabelText("Open workspace panel");
    fireEvent.click(filesButton);
    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeTab).toBe("files");
  });

  test("toggles the files card closed on a second click", () => {
    setPanel({ isOpen: true, activeTab: "files" });
    render(<ContextPanelToggle sessionId={SESSION} />);

    fireEvent.click(screen.getByLabelText("Hide workspace files"));
    expect(panelState().isOpen).toBe(false);
  });

  test("swaps the files trigger for a popover while the artifacts tab owns the sidebar", async () => {
    setPanel({ isOpen: true, activeTab: "artifacts" });
    render(<ContextPanelToggle sessionId={SESSION} />);

    expect(screen.queryByLabelText("Open workspace panel")).toBeNull();
    expect(screen.queryByLabelText("Hide workspace files")).toBeNull();
    expect(await screen.findByLabelText("Workspace files")).toBeDefined();
  });

  test("swaps the files trigger for a popover while an artifact is open", async () => {
    setPanel({ activeArtifact: ARTIFACT });
    render(<ContextPanelToggle sessionId={SESSION} />);

    expect(await screen.findByLabelText("Workspace files")).toBeDefined();
  });

  test("falls back to the files button when no sessionId is available, even with an artifact open", () => {
    setPanel({ activeArtifact: ARTIFACT });
    render(<ContextPanelToggle sessionId={null} />);

    expect(screen.getByLabelText("Open workspace panel")).toBeDefined();
  });

  test("sidebar toggle closes the artifact preview via closeArtifactPanel", () => {
    setPanel({ activeArtifact: ARTIFACT, isOpen: true });
    render(<ContextPanelToggle sessionId={SESSION} />);

    fireEvent.click(screen.getByLabelText("Hide artifacts"));

    expect(panelState().activeArtifact).toBeNull();
  });

  test("sidebar toggle restores the last previewed artifact instead of opening the tabs view", () => {
    setPanel({ lastArtifact: ARTIFACT, isOpen: false });
    render(<ContextPanelToggle sessionId={SESSION} />);

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().activeArtifact).toEqual(ARTIFACT);
  });

  test("sidebar toggle opens the artifacts tab when there is no remembered artifact", () => {
    setPanel({ lastArtifact: null, isOpen: false });
    render(<ContextPanelToggle sessionId={SESSION} />);

    fireEvent.click(screen.getByLabelText("Open artifacts"));

    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeTab).toBe("artifacts");
  });

  test("sidebar toggle reads as active when the artifacts tab is open", () => {
    setPanel({ isOpen: true, activeTab: "artifacts" });
    render(<ContextPanelToggle sessionId={SESSION} />);

    expect(screen.getByLabelText("Hide artifacts")).toBeDefined();
  });
});
