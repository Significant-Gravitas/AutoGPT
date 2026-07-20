import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Icon as PhosphorIcon, IconProps } from "@phosphor-icons/react";
import { forwardRef } from "react";

vi.mock("./agptIcons", () => ({
  getAutoGPTIcon: vi.fn(),
}));

import { createIcon } from "./createIcon";
import { getAutoGPTIcon, type AutoGPTIconProps } from "./agptIcons";
import { resetAutoGPTIconsAvailabilityCache } from "./helpers";

const mockedGetAutoGPTIcon = vi.mocked(getAutoGPTIcon);

function makeAgptIcon(exportName: string) {
  return function AgptIcon({ ariaLabel, ...props }: AutoGPTIconProps) {
    return (
      <svg
        data-testid="agpt-icon"
        data-export={exportName}
        aria-label={ariaLabel}
        {...props}
      />
    );
  };
}

const PhosphorDummy = forwardRef<SVGSVGElement, IconProps>(
  function PhosphorDummy({ weight }, ref) {
    return <svg ref={ref} data-testid="phosphor-icon" data-weight={weight} />;
  },
) as PhosphorIcon;

describe("createIcon", () => {
  beforeEach(() => {
    resetAutoGPTIconsAvailabilityCache();
    mockedGetAutoGPTIcon.mockImplementation((name) => makeAgptIcon(name));
  });

  it("renders the stroke AutoGPT icon by default", () => {
    const Heart = createIcon("HeartStroke", PhosphorDummy);

    render(<Heart />);

    expect(screen.getByTestId("agpt-icon").getAttribute("data-export")).toBe(
      "HeartStroke",
    );
  });

  it('swaps to the solid variant for weight="fill" when mapped', () => {
    const Heart = createIcon("HeartStroke", PhosphorDummy);

    render(<Heart weight="fill" />);

    expect(screen.getByTestId("agpt-icon").getAttribute("data-export")).toBe(
      "HeartSolid",
    );
  });

  it('keeps the stroke variant for weight="fill" without a solid mapping', () => {
    const Square = createIcon("SquareStroke", PhosphorDummy);

    render(<Square weight="fill" />);

    expect(screen.getByTestId("agpt-icon").getAttribute("data-export")).toBe(
      "SquareStroke",
    );
  });

  it("renders decoratively without an explicit label", () => {
    const Heart = createIcon("HeartStroke", PhosphorDummy);

    render(<Heart />);

    const icon = screen.getByTestId("agpt-icon");
    expect(icon.getAttribute("aria-hidden")).toBe("true");
    expect(icon.getAttribute("aria-label")).toBeNull();
  });

  it("forwards an explicit aria-label as the AutoGPT ariaLabel", () => {
    const Heart = createIcon("HeartStroke", PhosphorDummy);

    render(<Heart aria-label="Favorite" />);

    expect(screen.getByTestId("agpt-icon").getAttribute("aria-label")).toBe(
      "Favorite",
    );
  });

  it("falls back to the Phosphor icon with weight intact when unavailable", () => {
    mockedGetAutoGPTIcon.mockReturnValue(undefined);
    const Heart = createIcon("HeartStroke", PhosphorDummy);

    render(<Heart weight="fill" />);

    const icon = screen.getByTestId("phosphor-icon");
    expect(icon.getAttribute("data-weight")).toBe("fill");
  });
});
