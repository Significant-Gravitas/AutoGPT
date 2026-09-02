import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Only the artifacts flag resolves — querying any other flag returns undefined,
// so the trigger disappears and the assertions below fail loudly.
const { artifactsEnabledMock, useGetFlagMock, toggleContextPanelMock } =
  vi.hoisted(() => {
    const artifactsEnabledMock = { current: true };
    return {
      artifactsEnabledMock,
      useGetFlagMock: vi.fn((flag: string) =>
        flag === "artifacts" ? artifactsEnabledMock.current : undefined,
      ),
      toggleContextPanelMock: vi.fn(),
    };
  });

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

import { Flag } from "@/services/feature-flags/use-get-flag";
import { WorkspaceFilesTrigger } from "../WorkspaceFilesTrigger";

describe("WorkspaceFilesTrigger", () => {
  beforeEach(() => {
    toggleContextPanelMock.mockClear();
    useGetFlagMock.mockClear();
    artifactsEnabledMock.current = true;
  });

  it("gates on the artifacts flag specifically", () => {
    render(<WorkspaceFilesTrigger />);

    expect(useGetFlagMock).toHaveBeenCalledWith(Flag.ARTIFACTS);
  });

  it("toggles the context panel when clicked", () => {
    render(<WorkspaceFilesTrigger />);

    fireEvent.click(
      screen.getByRole("button", { name: /open workspace files/i }),
    );

    expect(toggleContextPanelMock).toHaveBeenCalledTimes(1);
  });

  it("renders nothing when the artifacts flag is off", () => {
    artifactsEnabledMock.current = false;

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
