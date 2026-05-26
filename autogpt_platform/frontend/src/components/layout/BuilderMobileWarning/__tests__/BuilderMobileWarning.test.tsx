import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BuilderMobileWarning } from "../BuilderMobileWarning";

let mockBreakpoint = "lg";

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => mockBreakpoint,
}));

afterEach(() => {
  mockBreakpoint = "lg";
});

describe("BuilderMobileWarning", () => {
  it("does not render on desktop breakpoints", () => {
    render(<BuilderMobileWarning />);

    expect(
      screen.queryByText(/builder requires a desktop browser/i),
    ).toBeNull();
  });

  it("renders on small mobile breakpoints", () => {
    mockBreakpoint = "sm";

    render(<BuilderMobileWarning />);

    expect(
      screen.getByText(/builder requires a desktop browser/i),
    ).toBeDefined();
    expect(
      screen.getByText(/graph builder uses canvas interactions/i),
    ).toBeDefined();
  });

  it("can be dismissed without blocking the page", () => {
    mockBreakpoint = "base";

    render(<BuilderMobileWarning />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss warning/i }));

    expect(
      screen.queryByText(/builder requires a desktop browser/i),
    ).toBeNull();
  });
});
