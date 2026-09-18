import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Key, storage } from "@/services/storage/local-storage";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelToggle } from "../ContextPanelToggle";

vi.mock("../components/FilesTab/useSessionFiles", () => ({
  useSessionFiles: () => ({ generated: [], uploaded: [] }),
}));

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
