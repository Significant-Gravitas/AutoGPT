import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Input } from "./Input";

describe("Input Component", () => {
  it("renders input with label", () => {
    render(<Input label="Username" />);

    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByText("Username")).toBeInTheDocument();
  });

  it("renders input with hidden label", () => {
    render(<Input label="Username" hideLabel />);

    const input = screen.getByRole("textbox");
    expect(input).toHaveAttribute("aria-label", "Username");
    expect(screen.queryByText("Username")).not.toBeInTheDocument();
  });

  it("calls onChange when typing", async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(<Input label="Username" onChange={handleChange} />);

    const input = screen.getByLabelText("Username");
    await user.type(input, "test");

    expect(handleChange).toHaveBeenCalled();
    expect(input).toHaveValue("test");
  });

  it("displays placeholder text", () => {
    render(<Input label="Username" placeholder="Enter your username" />);

    const input = screen.getByLabelText("Username");
    expect(input).toHaveAttribute("placeholder", "Enter your username");
  });

  it("uses label as placeholder when no placeholder provided", () => {
    render(<Input label="Username" />);

    const input = screen.getByLabelText("Username");
    expect(input).toHaveAttribute("placeholder", "Username");
  });

  it("displays error message", () => {
    render(<Input label="Username" error="Username is required" />);

    expect(screen.getByText("Username is required")).toBeInTheDocument();
    const input = screen.getByRole("textbox");
    expect(input).toBeInTheDocument();
  });

  it("is disabled when disabled prop is true", () => {
    render(<Input label="Username" disabled />);

    const input = screen.getByLabelText("Username");
    expect(input).toBeDisabled();
  });

  it("does not call onChange when disabled", async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(<Input label="Username" disabled onChange={handleChange} />);

    const input = screen.getByLabelText("Username");
    await user.type(input, "test");

    expect(handleChange).not.toHaveBeenCalled();
    expect(input).toHaveValue("");
  });

  it("supports different input types", () => {
    const { rerender } = render(<Input label="Email" type="email" />);
    expect(screen.getByLabelText("Email")).toHaveAttribute("type", "email");

    rerender(<Input label="Password" type="password" />);
    expect(screen.getByLabelText("Password")).toHaveAttribute(
      "type",
      "password",
    );

    rerender(<Input label="Number" type="number" />);
    expect(screen.getByLabelText("Number")).toHaveAttribute("type", "number");
  });

  it("handles focus and blur events", async () => {
    const handleFocus = vi.fn();
    const handleBlur = vi.fn();
    const user = userEvent.setup();

    render(
      <Input label="Username" onFocus={handleFocus} onBlur={handleBlur} />,
    );

    const input = screen.getByLabelText("Username");

    await user.click(input);
    expect(handleFocus).toHaveBeenCalledTimes(1);

    await user.tab();
    expect(handleBlur).toHaveBeenCalledTimes(1);
  });

  it("accepts a default value", () => {
    render(<Input label="Username" defaultValue="john_doe" />);

    const input = screen.getByLabelText("Username");
    expect(input).toHaveValue("john_doe");
  });

  it("can be controlled with value prop", async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    const { rerender } = render(
      <Input label="Username" value="initial" onChange={handleChange} />,
    );

    const input = screen.getByLabelText("Username");
    expect(input).toHaveValue("initial");

    // Try typing - should call onChange but value stays controlled
    await user.type(input, "x");
    expect(handleChange).toHaveBeenCalled();

    // Update with new controlled value
    rerender(
      <Input label="Username" value="updated" onChange={handleChange} />,
    );
    expect(input).toHaveValue("updated");
  });

  it("supports keyboard navigation", async () => {
    const user = userEvent.setup();

    render(<Input label="Username" />);

    const input = screen.getByLabelText("Username");

    // Tab to focus
    await user.tab();
    expect(input).toHaveFocus();

    // Type some text
    await user.keyboard("test");
    expect(input).toHaveValue("test");

    // Navigate within text
    await user.keyboard("{Home}");
    await user.keyboard("start");
    expect(input).toHaveValue("starttest");
  });
});
