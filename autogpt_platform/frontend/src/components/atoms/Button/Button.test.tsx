import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Button } from "./Button";

describe("Button Component", () => {
  it("renders button with text", () => {
    render(<Button>Click me</Button>);
    expect(
      screen.getByRole("button", { name: "Click me" }),
    ).toBeInTheDocument();
  });

  it("calls onClick when clicked", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={handleClick}>Click me</Button>);

    await user.click(screen.getByRole("button"));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it("is disabled when disabled prop is true", () => {
    render(<Button disabled>Disabled button</Button>);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("does not call onClick when disabled", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(
      <Button disabled onClick={handleClick}>
        Disabled button
      </Button>,
    );

    // Try to click the disabled button
    await user.click(screen.getByRole("button"));
    expect(handleClick).not.toHaveBeenCalled();
  });

  it("renders with different variants", () => {
    const { rerender } = render(<Button variant="primary">Primary</Button>);
    expect(screen.getByRole("button")).toHaveClass("bg-zinc-800");

    rerender(<Button variant="secondary">Secondary</Button>);
    expect(screen.getByRole("button")).toHaveClass("bg-zinc-100");

    rerender(<Button variant="destructive">Destructive</Button>);
    expect(screen.getByRole("button")).toHaveClass("bg-red-500");

    rerender(<Button variant="outline">Outline</Button>);
    expect(screen.getByRole("button")).toHaveClass("border-zinc-700");

    rerender(<Button variant="ghost">Ghost</Button>);
    expect(screen.getByRole("button")).toHaveClass("bg-transparent");
  });

  // Sizes and variants styling are covered by Chromatic via Storybook

  it("shows loading state", () => {
    render(<Button loading>Loading button</Button>);

    const button = screen.getByRole("button");
    expect(button).toHaveClass("pointer-events-none");

    // Check for loading spinner (svg element)
    const spinner = button.querySelector("svg");
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass("animate-spin");
  });

  it("renders with left icon", () => {
    const TestIcon = () => <span data-testid="test-icon">Icon</span>;

    render(<Button leftIcon={<TestIcon />}>Button with left icon</Button>);

    expect(screen.getByTestId("test-icon")).toBeInTheDocument();
    expect(screen.getByText("Button with left icon")).toBeInTheDocument();
  });

  it("renders with right icon", () => {
    const TestIcon = () => <span data-testid="test-icon">Icon</span>;

    render(<Button rightIcon={<TestIcon />}>Button with right icon</Button>);

    expect(screen.getByTestId("test-icon")).toBeInTheDocument();
    expect(screen.getByText("Button with right icon")).toBeInTheDocument();
  });

  it("supports keyboard navigation", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={handleClick}>Keyboard button</Button>);

    const button = screen.getByRole("button");

    // Focus the button
    await user.tab();
    expect(button).toHaveFocus();

    // Press Enter
    await user.keyboard("{Enter}");
    expect(handleClick).toHaveBeenCalledTimes(1);

    // Press Space
    await user.keyboard(" ");
    expect(handleClick).toHaveBeenCalledTimes(2);
  });

  it("applies custom className", () => {
    render(<Button className="custom-class">Custom button</Button>);
    expect(screen.getByRole("button")).toHaveClass("custom-class");
  });

  it("handles double click", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={handleClick}>Double click me</Button>);

    await user.dblClick(screen.getByRole("button"));
    expect(handleClick).toHaveBeenCalledTimes(2);
  });

  it("maintains focus after click", async () => {
    const user = userEvent.setup();

    render(<Button>Focus test</Button>);

    const button = screen.getByRole("button");
    await user.click(button);
    expect(button).toHaveFocus();
  });
});
