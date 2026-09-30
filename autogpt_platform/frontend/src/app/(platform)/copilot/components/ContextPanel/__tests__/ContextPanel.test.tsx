import { beforeEach, describe, expect, test, vi } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { getListExpertCredentialsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { useCopilotUIStore } from "../../../store";
import { ContextPanel } from "../ContextPanel";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
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
    contextPanelExpert: null,
    artifactPanel: {
      ...s.artifactPanel,
      isOpen: true,
      activeArtifact: null,
      activeTab: "artifacts",
      mode: "artifact",
      isComputerOpen: false,
    },
  }));
});

describe("ContextPanel", () => {
  test("docks for the artifacts tab and renders the artifacts library", async () => {
    const { container } = render(<ContextPanel sessionId="session-1" />);
    await waitFor(() =>
      expect(container.querySelector("[data-context-panel]")).not.toBeNull(),
    );
    expect(await screen.findByText("Nothing to preview yet.")).toBeDefined();
  });

  test("docks for the files tab and lists the chat's files", async () => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, activeTab: "files" },
    }));
    const { container } = render(<ContextPanel sessionId="session-1" />);
    await waitFor(() =>
      expect(container.querySelector("[data-context-panel]")).not.toBeNull(),
    );
    expect(await screen.findByText("Nothing here yet.")).toBeDefined();
  });

  test("docks for the integrations tab with the expert's integrations", async () => {
    server.use(getListExpertCredentialsMockHandler([]));
    useCopilotUIStore.setState((s) => ({
      contextPanelExpert: { id: "expert-maria", name: "Maria" },
      artifactPanel: { ...s.artifactPanel, activeTab: "integrations" },
    }));
    render(<ContextPanel sessionId="session-1" />);

    expect(await screen.findByText("Maria's Integrations")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Open Maria's page" }),
    ).toBeDefined();
  });

  test("hides itself while an artifact is previewing (artifact takes over the region)", () => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: {
        ...s.artifactPanel,
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
  });

  test("hides itself while the computer face is showing, whatever tab is remembered", () => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: {
        ...s.artifactPanel,
        activeTab: "artifacts",
        mode: "computer",
        isComputerOpen: true,
      },
    }));
    const { container } = render(<ContextPanel sessionId="session-1" />);
    expect(container.querySelector("[data-context-panel]")).toBeNull();
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

  test("mobile: opens the sheet on the files tab", async () => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, activeTab: "files" },
    }));
    render(<ContextPanel sessionId="session-1" mobile />);
    expect(await screen.findByRole("dialog")).toBeDefined();
    expect(screen.getByRole("heading", { name: "Files" })).toBeDefined();
  });

  test("mobile: opens the sheet on the artifacts tab", async () => {
    render(<ContextPanel sessionId="session-1" mobile />);
    expect(await screen.findByRole("dialog")).toBeDefined();
    expect(screen.getByRole("heading", { name: "Artifacts" })).toBeDefined();
  });
});
