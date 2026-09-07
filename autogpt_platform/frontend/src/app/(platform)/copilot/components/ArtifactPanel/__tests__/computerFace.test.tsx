import {
  getGetV2GetSessionComputerMockHandler,
  getPostV2StartSessionDesktopMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ComputerPanelContent } from "../components/ComputerPanelContent";
import { PanelModeSwitch } from "../components/PanelModeSwitch";

const STREAM = {
  url: "https://6080-sbx.e2b.app/vnc.html?autoconnect=true",
  sandbox_id: "sbx-1",
  provider: "e2b" as const,
};

describe("copilot store: computer face", () => {
  beforeEach(() => {
    useCopilotUIStore.getState().resetArtifactPanel();
    useCopilotUIStore.getState().closeArtifactPanel({ persist: false });
  });

  it("a started desktop opens the computer face when nothing else is showing", () => {
    useCopilotUIStore.getState().registerComputerStream(STREAM);
    const panel = useCopilotUIStore.getState().artifactPanel;
    expect(panel.computer?.sandbox_id).toBe("sbx-1");
    expect(panel.mode).toBe("computer");
    expect(panel.isComputerOpen).toBe(true);
  });

  it("does not steal the panel from an open artifact, but remembers the desktop", () => {
    useCopilotUIStore.getState().openArtifact(
      {
        id: "a1",
        title: "notes.md",
        sourceUrl: "/x",
        origin: "assistant",
      } as never,
      { persist: false },
    );
    useCopilotUIStore.getState().registerComputerStream(STREAM);
    const panel = useCopilotUIStore.getState().artifactPanel;
    expect(panel.mode).toBe("artifact");
    expect(panel.computer?.sandbox_id).toBe("sbx-1");
    useCopilotUIStore.getState().setArtifactPanelMode("computer");
    expect(useCopilotUIStore.getState().artifactPanel.isComputerOpen).toBe(
      true,
    );
  });

  it("closing the panel and entering a new chat both drop the computer face", () => {
    useCopilotUIStore.getState().registerComputerStream(STREAM);
    useCopilotUIStore.getState().closeArtifactPanel({ persist: false });
    expect(useCopilotUIStore.getState().artifactPanel.isComputerOpen).toBe(
      false,
    );
    useCopilotUIStore.getState().clearLastArtifact();
    expect(useCopilotUIStore.getState().artifactPanel.computer).toBeNull();
  });
});

describe("PanelModeSwitch", () => {
  afterEach(() => cleanup());

  it("marks the active face and disables Artifact when there is none", async () => {
    const changes: string[] = [];
    render(
      <PanelModeSwitch
        mode="computer"
        hasArtifact={false}
        onChange={(m) => changes.push(m)}
      />,
    );
    const computer = screen.getByRole("button", { name: "Computer" });
    const artifact = screen.getByRole("button", { name: "Artifact" });
    expect(computer.getAttribute("aria-pressed")).toBe("true");
    expect(artifact).toHaveProperty("disabled", true);
    await userEvent.click(computer);
    expect(changes).toEqual(["computer"]);
  });
});

describe("ComputerPanelContent", () => {
  beforeEach(() => {
    useCopilotUIStore.getState().resetArtifactPanel();
  });
  afterEach(() => cleanup());

  it("shows the chat's boxes and starts a desktop on demand", async () => {
    server.use(
      getGetV2GetSessionComputerMockHandler({
        owner_kind: "expert",
        owner_id: "exp-1",
        e2b_active: true,
        shell: {
          kind: "shell",
          sandbox_id: "sb",
          state: "paused",
          started_at: new Date("2026-09-05T12:00:00Z"),
          cpu_count: 2,
          memory_mb: 512,
          template_id: "base",
          mounts_attached: true,
        },
        desktop: null,
        mounts: {},
        workspace_path: "/home/user/workspace",
        shared_path: "/home/user/shared",
      }),
      getPostV2StartSessionDesktopMockHandler({
        kind: "desktop_stream",
        requires_auth: false,
        ...STREAM,
      }),
    );

    render(<ComputerPanelContent sessionId="sess-1" />);

    expect(await screen.findByText("Suspended")).toBeDefined();
    expect(screen.getByText(/expert's own computer/)).toBeDefined();
    await userEvent.click(
      screen.getByRole("button", { name: "Start desktop" }),
    );
    await waitFor(
      () =>
        expect(
          useCopilotUIStore.getState().artifactPanel.computer?.sandbox_id,
        ).toBe("sbx-1"),
      { timeout: 4000 },
    );
    await waitFor(() =>
      expect(screen.getByTitle("Interactive desktop (sbx-1)")).toBeDefined(),
    );
  });
});
