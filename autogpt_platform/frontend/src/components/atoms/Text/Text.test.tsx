import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Text } from "./Text";

describe("Text Component", () => {
  it("renders text content", () => {
    render(<Text variant="body">Hello World</Text>);

    expect(screen.getByText("Hello World")).toBeInTheDocument();
  });

  it("renders with different variants", () => {
    const { rerender } = render(<Text variant="body">Body Text</Text>);
    expect(screen.getByText("Body Text")).toBeInTheDocument();

    rerender(<Text variant="body-medium">Medium Text</Text>);
    expect(screen.getByText("Medium Text")).toBeInTheDocument();

    rerender(<Text variant="small">Small Text</Text>);
    expect(screen.getByText("Small Text")).toBeInTheDocument();
  });

  it("renders with custom element using as prop", () => {
    render(
      <Text variant="body" as="h1">
        Heading Text
      </Text>,
    );

    const element = screen.getByText("Heading Text");
    expect(element.tagName).toBe("H1");
  });

  it("applies custom className", () => {
    render(
      <Text variant="body" className="custom-text">
        Styled Text
      </Text>,
    );

    const element = screen.getByText("Styled Text");
    expect(element).toHaveClass("custom-text");
  });

  it("passes through HTML attributes", () => {
    render(
      <Text variant="body" data-testid="text-element" title="Tooltip text">
        Text with attributes
      </Text>,
    );

    const element = screen.getByTestId("text-element");
    expect(element).toHaveAttribute("title", "Tooltip text");
  });

  it("supports onClick handler", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(
      <Text variant="body" onClick={handleClick}>
        Clickable Text
      </Text>,
    );

    const element = screen.getByText("Clickable Text");
    await user.click(element);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it("renders children correctly", () => {
    render(
      <Text variant="body">
        <span>Child 1</span>
        <span>Child 2</span>
      </Text>,
    );

    expect(screen.getByText("Child 1")).toBeInTheDocument();
    expect(screen.getByText("Child 2")).toBeInTheDocument();
  });

  it("supports different text variants", () => {
    const variants = [
      "h1",
      "h2",
      "h3",
      "h4",
      "lead",
      "large",
      "large-medium",
      "large-semibold",
      "body",
      "body-medium",
      "small",
      "small-medium",
      "subtle",
    ];

    variants.forEach((variant) => {
      const { unmount } = render(
        <Text variant={variant as any} data-testid={`text-${variant}`}>
          {variant} text
        </Text>,
      );

      expect(screen.getByTestId(`text-${variant}`)).toBeInTheDocument();
      unmount();
    });
  });

  it("handles empty children", () => {
    render(<Text variant="body" data-testid="empty-text"></Text>);

    const element = screen.getByTestId("empty-text");
    expect(element).toBeInTheDocument();
    expect(element).toBeEmptyDOMElement();
  });

  it("supports keyboard navigation when interactive", async () => {
    const handleKeyDown = vi.fn();
    const user = userEvent.setup();

    render(
      <Text variant="body" tabIndex={0} onKeyDown={handleKeyDown}>
        Interactive Text
      </Text>,
    );

    const element = screen.getByText("Interactive Text");

    // Focus with tab
    await user.tab();
    expect(element).toHaveFocus();

    // Press key
    await user.keyboard("{Enter}");
    expect(handleKeyDown).toHaveBeenCalled();
  });
});
