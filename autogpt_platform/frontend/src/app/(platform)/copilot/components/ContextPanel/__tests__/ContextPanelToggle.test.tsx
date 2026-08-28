import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelToggle } from "../ContextPanelToggle";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
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
  setPanel({
    isOpen: false,
    activeArtifact: null,
    activeTab: "files",
    lastArtifact: null,
    history: [],
  });
});

describe("ContextPanelToggle", () => {
  test("shows a files toggle button when the right sidebar is closed", () => {
    render(<ContextPanelToggle sessionId={SESSION} />);

    const filesButton = screen.getByLabelText("Open workspace files");
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

    expect(screen.queryByLabelText("Open workspace files")).toBeNull();
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

    expect(screen.getByLabelText("Open workspace files")).toBeDefined();
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
