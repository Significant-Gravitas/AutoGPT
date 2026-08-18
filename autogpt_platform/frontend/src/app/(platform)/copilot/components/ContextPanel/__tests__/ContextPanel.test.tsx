import { beforeEach, describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { useCopilotUIStore } from "../../../store";
import { ContextPanel } from "../ContextPanel";

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
