import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { useCopilotUIStore } from "../../../store";
import { ContextPanel } from "../ContextPanel";

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

afterEach(() => {
  flagState.newToolUI = false;
});

beforeEach(() => {
  server.use(
    getListWorkspaceFilesMockHandler200({
      files: [],
      offset: 0,
      has_more: false,
    }),
  );
  useCopilotUIStore.setState((s) => ({
    artifactPanel: {
      ...s.artifactPanel,
      isOpen: true,
      activeArtifact: null,
      activeTab: "files",
    },
  }));
});

describe("ContextPanel", () => {
  test("renders the tab switcher and active tab when open with no artifact", async () => {
    render(<ContextPanel sessionId="session-1" />);
    expect(await screen.findByRole("tablist")).toBeDefined();
    expect(screen.getByRole("tab", { name: /^Files \(\d+\)$/ })).toBeDefined();
    expect(await screen.findByText(/No files yet/i)).toBeDefined();
  });

  test("hides itself while an artifact is previewing (artifact takes over the region)", () => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: {
        ...s.artifactPanel,
        isOpen: true,
        activeArtifact: {
          id: "f1",
          title: "doc.md",
          mimeType: "text/markdown",
          sourceUrl: "/api/proxy/api/workspace/files/f1/download",
          origin: "agent",
        },
      },
    }));
    const { container } = render(<ContextPanel sessionId="session-1" />);
    expect(container.querySelector("[data-context-panel]")).toBeNull();
    expect(screen.queryByRole("tablist")).toBeNull();
  });

  test("leaves a files tab alone and stays undocked so the files card owns it", async () => {
    flagState.newToolUI = true;
    const { container } = render(<ContextPanel sessionId="session-1" />);
    await waitFor(() =>
      expect(container.querySelector("[data-context-panel]")).toBeNull(),
    );
    expect(useCopilotUIStore.getState().artifactPanel.activeTab).toBe("files");
  });

  test("docks for the artifacts tab under the new tool UI", async () => {
    flagState.newToolUI = true;
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, activeTab: "artifacts" },
    }));
    const { container } = render(<ContextPanel sessionId="session-1" />);
    await waitFor(() =>
      expect(container.querySelector("[data-context-panel]")).not.toBeNull(),
    );
    expect(useCopilotUIStore.getState().artifactPanel.activeTab).toBe(
      "artifacts",
    );
  });

  test("renders nothing when closed", () => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: {
        ...s.artifactPanel,
        isOpen: false,
        activeArtifact: null,
      },
    }));
    const { container } = render(<ContextPanel sessionId="session-1" />);
    expect(container.querySelector("[data-context-panel]")).toBeNull();
  });
});
