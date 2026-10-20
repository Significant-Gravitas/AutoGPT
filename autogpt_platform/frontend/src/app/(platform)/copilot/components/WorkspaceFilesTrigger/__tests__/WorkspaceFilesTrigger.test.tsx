import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { MobileHeader } from "../../MobileHeader/MobileHeader";
import { WorkspaceFilesTrigger } from "../WorkspaceFilesTrigger";

beforeEach(() => {
  useCopilotUIStore.setState((state) => ({
    contextPanelExpert: null,
    artifactPanel: {
      ...state.artifactPanel,
      isOpen: false,
      activeTab: "files",
      activeArtifact: null,
      lastArtifact: null,
      history: [],
      mode: "artifact",
      isComputerOpen: false,
    },
  }));
});

describe("WorkspaceFilesTrigger", () => {
  it("offers workspace files without a feature flag", () => {
    render(<WorkspaceFilesTrigger />);
    expect(
      screen.getByRole("button", { name: /open workspace files/i }),
    ).toBeDefined();
  });

  it("toggles the context panel when clicked", () => {
    render(<WorkspaceFilesTrigger />);

    fireEvent.click(
      screen.getByRole("button", { name: /open workspace files/i }),
    );

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });

  it("forwards className so the header can restyle it", () => {
    render(<WorkspaceFilesTrigger className="rounded-full" />);

    expect(
      screen.getByRole("button", { name: /open workspace files/i }).className,
    ).toContain("rounded-full");
  });
});

describe.each([
  ["classic layout", MobileHeader],
  ["new layout", WorkspaceFilesTrigger],
] as const)("workspace files button in the %s", (_layout, Controls) => {
  it.each([true, false])(
    "opens files after integrations when the panel is open: %s",
    (isOpen) => {
      useCopilotUIStore.getState().toggleIntegrationsPanel({
        id: "expert-maria",
        name: "Maria",
      });
      if (!isOpen)
        useCopilotUIStore.getState().toggleContextPanelTab("integrations");
      render(<Controls />);

      fireEvent.click(
        screen.getByRole("button", { name: /open workspace files/i }),
      );

      expect(useCopilotUIStore.getState().artifactPanel).toMatchObject({
        isOpen: true,
        activeTab: "files",
      });
    },
  );

  it("closes files when clicked again", () => {
    render(<Controls />);
    const button = screen.getByRole("button", {
      name: /open workspace files/i,
    });

    fireEvent.click(button);
    fireEvent.click(button);

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });
});
