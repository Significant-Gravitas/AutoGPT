import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MobileWarning } from "../MobileWarning";

const mockUseBreakpoint = vi.fn();

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => mockUseBreakpoint(),
}));

describe("MobileWarning", () => {
  beforeEach(() => {
    mockUseBreakpoint.mockReturnValue("sm");
  });

  afterEach(() => {
    mockUseBreakpoint.mockReset();
  });

  it("renders the warning on mobile breakpoints", () => {
    render(<MobileWarning />);
    expect(screen.getByText(/builder works best on desktop/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /continue anyway/i }),
    ).toBeDefined();
  });

  it("renders nothing on large breakpoints", () => {
    mockUseBreakpoint.mockReturnValue("lg");
    const { container } = render(<MobileWarning />);
    expect(container.innerHTML).toBe("");
  });

  it("dismisses when the user clicks 'Continue anyway'", () => {
    render(<MobileWarning />);
    fireEvent.click(screen.getByRole("button", { name: /continue anyway/i }));
    expect(screen.queryByText(/builder works best on desktop/i)).toBeNull();
  });
});
