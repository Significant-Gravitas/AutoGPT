import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelToggle } from "../ContextPanelToggle";

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

describe("ContextPanelToggle files button", () => {
  test("opens the files tab of the side panel", () => {
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open files"));

    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeTab).toBe("files");
    expect(screen.getByLabelText("Hide files")).toBeDefined();
  });

  test("closes the files tab when it is showing", () => {
    setPanel({ isOpen: true, activeTab: "files" });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Hide files"));

    expect(panelState().isOpen).toBe(false);
  });

  test("brings the files tab back over an open preview", () => {
    setPanel({ activeArtifact: ARTIFACT, isOpen: true });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open files"));

    expect(panelState().activeArtifact).toBeNull();
    expect(panelState().isOpen).toBe(true);
    expect(panelState().activeTab).toBe("files");
  });

  test("takes the panel from the computer face", () => {
    setPanel({ isOpen: true, mode: "computer", isComputerOpen: true });
    render(<ContextPanelToggle sessionId="s1" />);

    fireEvent.click(screen.getByLabelText("Open files"));

    expect(panelState().isComputerOpen).toBe(false);
    expect(panelState().mode).toBe("artifact");
    expect(panelState().activeTab).toBe("files");
  });

  test("shows no integrations without an expert", () => {
    render(<ContextPanelToggle sessionId="s1" />);

    expect(screen.queryByTestId("expert-integrations")).toBeNull();
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
    expect(screen.getByLabelText("Open files")).toBeDefined();
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

  test("is hidden on mobile, whose sheet has no computer face", () => {
    setMobile(true);
    render(<ContextPanelToggle sessionId="s1" />);

    expect(screen.queryByLabelText("Open computer")).toBeNull();
    expect(screen.getByLabelText("Open files")).toBeDefined();
  });
});
