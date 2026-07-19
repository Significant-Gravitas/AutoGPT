import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./agptIcons", () => ({
  getAutoGPTIcon: vi.fn(),
}));

import { Icon } from "./Icon";
import { getAutoGPTIcon, type AutoGPTIconProps } from "./agptIcons";

const mockedGetAutoGPTIcon = vi.mocked(getAutoGPTIcon);

describe("Icon", () => {
  it("renders the AutoGPT icon when the optional package provides it", () => {
    mockedGetAutoGPTIcon.mockReturnValue(function AgptIcon({
      ariaLabel,
    }: AutoGPTIconProps) {
      return <svg data-testid="agpt-icon" aria-label={ariaLabel} />;
    });

    render(<Icon name="home" aria-label="Home" />);

    const icon = screen.getByTestId("agpt-icon");
    expect(icon).toBeTruthy();
    expect(icon.getAttribute("aria-label")).toBe("Home");
  });

  it("falls back to the Phosphor icon when the package is absent", () => {
    mockedGetAutoGPTIcon.mockReturnValue(undefined);

    const { container } = render(<Icon name="home" aria-label="Home" />);

    expect(screen.queryByTestId("agpt-icon")).toBeNull();
    const svg = container.querySelector("svg");
    expect(svg).toBeTruthy();
    expect(svg?.getAttribute("aria-label")).toBe("Home");
  });

  it("uses the icon name as the default accessible label", () => {
    mockedGetAutoGPTIcon.mockReturnValue(undefined);

    const { container } = render(<Icon name="search" />);

    expect(container.querySelector("svg")?.getAttribute("aria-label")).toBe(
      "search",
    );
  });
});
