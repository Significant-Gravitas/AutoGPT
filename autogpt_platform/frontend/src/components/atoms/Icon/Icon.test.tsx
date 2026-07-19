import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./agptIcons", () => ({
  getAutoGPTIcon: vi.fn(),
}));

import { Icon } from "./Icon";
import { getAutoGPTIcon, type AutoGPTIconProps } from "./agptIcons";

const mockedGetAutoGPTIcon = vi.mocked(getAutoGPTIcon);

function AgptIcon({ ariaLabel }: AutoGPTIconProps) {
  return <svg data-testid="agpt-icon" aria-label={ariaLabel} />;
}

describe("Icon", () => {
  it("renders AutoGPT icons when every registry icon resolves", () => {
    mockedGetAutoGPTIcon.mockReturnValue(AgptIcon);

    render(<Icon name="home" aria-label="Home" />);

    const icon = screen.getByTestId("agpt-icon");
    expect(icon).toBeTruthy();
    expect(icon.getAttribute("aria-label")).toBe("Home");
  });

  it("falls back to Phosphor when the package is absent", () => {
    mockedGetAutoGPTIcon.mockReturnValue(undefined);

    const { container } = render(<Icon name="home" aria-label="Home" />);

    expect(screen.queryByTestId("agpt-icon")).toBeNull();
    const svg = container.querySelector("svg");
    expect(svg).toBeTruthy();
    expect(svg?.getAttribute("aria-label")).toBe("Home");
  });

  it("never mixes: if any AutoGPT icon is missing, all render as Phosphor", () => {
    // Every icon resolves EXCEPT one — so the whole app must use Phosphor, even
    // for an icon whose AutoGPT equivalent would have resolved.
    mockedGetAutoGPTIcon.mockImplementation((name) =>
      name === "HomeDefaultStroke" ? undefined : AgptIcon,
    );

    // "search" -> "SearchDefaultStroke" resolves, but availability is false.
    const { container } = render(<Icon name="search" aria-label="Search" />);

    expect(screen.queryByTestId("agpt-icon")).toBeNull();
    const svg = container.querySelector("svg");
    expect(svg).toBeTruthy();
    expect(svg?.getAttribute("aria-label")).toBe("Search");
  });

  it("uses the icon name as the default accessible label", () => {
    mockedGetAutoGPTIcon.mockReturnValue(undefined);

    const { container } = render(<Icon name="search" />);

    expect(container.querySelector("svg")?.getAttribute("aria-label")).toBe(
      "search",
    );
  });
});
