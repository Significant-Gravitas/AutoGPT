import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { useGetFlagMock, toggleContextPanelMock } = vi.hoisted(() => ({
  useGetFlagMock: vi.fn(() => true),
  toggleContextPanelMock: vi.fn(),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: useGetFlagMock };
});

vi.mock("../../../store", () => ({
  useCopilotUIStore: (selector: (state: unknown) => unknown) =>
    selector({ toggleContextPanel: toggleContextPanelMock }),
}));

import { WorkspaceFilesTrigger } from "../WorkspaceFilesTrigger";

describe("WorkspaceFilesTrigger", () => {
  beforeEach(() => {
    toggleContextPanelMock.mockClear();
    useGetFlagMock.mockReturnValue(true);
  });

  it("toggles the context panel when clicked", () => {
    render(<WorkspaceFilesTrigger />);

    fireEvent.click(
      screen.getByRole("button", { name: /open workspace files/i }),
    );

    expect(toggleContextPanelMock).toHaveBeenCalledTimes(1);
  });

  it("renders nothing when the artifacts flag is off", () => {
    useGetFlagMock.mockReturnValue(false);

    render(<WorkspaceFilesTrigger />);

    expect(
      screen.queryByRole("button", { name: /open workspace files/i }),
    ).toBeNull();
  });

  it("forwards className so the header can restyle it", () => {
    render(<WorkspaceFilesTrigger className="rounded-full" />);

    expect(
      screen.getByRole("button", { name: /open workspace files/i }).className,
    ).toContain("rounded-full");
  });
});
