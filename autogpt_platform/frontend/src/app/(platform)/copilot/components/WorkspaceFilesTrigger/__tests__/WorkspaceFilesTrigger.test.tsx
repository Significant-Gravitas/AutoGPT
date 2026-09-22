import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { toggleContextPanelMock } = vi.hoisted(() => ({
  toggleContextPanelMock: vi.fn(),
}));

vi.mock("../../../store", () => ({
  useCopilotUIStore: (selector: (state: unknown) => unknown) =>
    selector({ toggleContextPanel: toggleContextPanelMock }),
}));

import { WorkspaceFilesTrigger } from "../WorkspaceFilesTrigger";

describe("WorkspaceFilesTrigger", () => {
  beforeEach(() => {
    toggleContextPanelMock.mockClear();
  });

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

    expect(toggleContextPanelMock).toHaveBeenCalledTimes(1);
  });

  it("forwards className so the header can restyle it", () => {
    render(<WorkspaceFilesTrigger className="rounded-full" />);

    expect(
      screen.getByRole("button", { name: /open workspace files/i }).className,
    ).toContain("rounded-full");
  });
});
