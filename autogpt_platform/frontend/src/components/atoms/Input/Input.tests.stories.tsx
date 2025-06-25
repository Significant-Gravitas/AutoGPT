import type { Meta, StoryObj } from "@storybook/nextjs";
import { expect, fn, userEvent, within } from "storybook/test";
import { Input } from "./Input";

const meta: Meta<typeof Input> = {
  title: "Atoms/Tests/Input",
  component: Input,
  parameters: {
    layout: "centered",
  },
  tags: ["!autodocs"],
  args: {
    onChange: fn(),
    onFocus: fn(),
    onBlur: fn(),
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Test basic input functionality
export const BasicInputInteraction: Story = {
  args: {
    label: "Username",
    placeholder: "Enter your username",
  },
  play: async function testBasicInput({ args, canvasElement }) {
    const canvas = within(canvasElement);
    const input = canvas.getByLabelText("Username");

    // Test initial state
    expect(input).toBeInTheDocument();
    expect(input).not.toBeDisabled();
    expect(input).toHaveValue("");

    // Test typing
    await userEvent.type(input, "test");
    expect(input).toHaveValue("test");

    // Test that onChange was called
    expect(args.onChange).toHaveBeenCalled();
  },
};

// Test input with hidden label
export const HiddenLabelInteraction: Story = {
  args: {
    label: "Search",
    placeholder: "Search...",
    hideLabel: true,
  },
  play: async function testHiddenLabel({ canvasElement }) {
    const canvas = within(canvasElement);
    const input = canvas.getByLabelText("Search");

    // Test accessibility with hidden label
    expect(input).toHaveAccessibleName("Search");
    expect(input).toHaveAttribute("aria-label", "Search");

    // Test functionality
    await userEvent.type(input, "query");
    expect(input).toHaveValue("query");
  },
};

// Test error state
export const ErrorStateInteraction: Story = {
  args: {
    label: "Password",
    placeholder: "Enter password",
    error: "Password is required",
  },
  play: async function testErrorState({ canvasElement }) {
    const canvas = within(canvasElement);
    const input = canvas.getByPlaceholderText("Enter password");
    const errorMessage = canvas.getByText("Password is required");

    // Test error state rendering
    expect(input).toBeInTheDocument();
    expect(errorMessage).toBeInTheDocument();

    // Test error styling
    expect(input).toHaveClass("border-red-500");
    expect(errorMessage).toHaveClass("!text-red-500");
  },
};

// Test different input types
export const InputTypesInteraction: Story = {
  render: function renderInputTypes() {
    return (
      <div className="space-y-4">
        <Input label="Text Input" type="text" />
        <Input label="Email Input" type="email" />
        <Input label="Password Input" type="password" />
        <Input label="Number Input" type="number" />
      </div>
    );
  },
  play: async function testInputTypes({ canvasElement }) {
    const canvas = within(canvasElement);

    // Test all input types
    const textInput = canvas.getByLabelText("Text Input");
    const emailInput = canvas.getByLabelText("Email Input");
    const passwordInput = canvas.getByLabelText("Password Input");
    const numberInput = canvas.getByLabelText("Number Input");

    // Test correct types are applied
    expect(textInput).toHaveAttribute("type", "text");
    expect(emailInput).toHaveAttribute("type", "email");
    expect(passwordInput).toHaveAttribute("type", "password");
    expect(numberInput).toHaveAttribute("type", "number");

    // Test that all inputs are rendered
    expect(textInput).toBeInTheDocument();
    expect(emailInput).toBeInTheDocument();
    expect(passwordInput).toBeInTheDocument();
    expect(numberInput).toBeInTheDocument();
  },
};

// Test disabled state
export const DisabledInteraction: Story = {
  args: {
    label: "Disabled Input",
    placeholder: "This is disabled",
    disabled: true,
  },
  play: async function testDisabledInput({ args, canvasElement }) {
    const canvas = within(canvasElement);
    const input = canvas.getByLabelText("Disabled Input");

    // Test disabled state
    expect(input).toBeDisabled();
    expect(input).toHaveAttribute("disabled");

    // Test that disabled input has empty value and onChange is not called
    // We don't test typing because disabled inputs prevent user interaction
    expect(input).toHaveValue("");
    expect(args.onChange).not.toHaveBeenCalled();
  },
};

// Test clear input functionality
export const ClearInputInteraction: Story = {
  args: {
    label: "Clear Test",
    placeholder: "Type and clear",
  },
  play: async function testClearInput({ canvasElement }) {
    const canvas = within(canvasElement);
    const input = canvas.getByLabelText("Clear Test");

    // Type something
    await userEvent.type(input, "content");
    expect(input).toHaveValue("content");

    // Clear with select all + delete
    await userEvent.keyboard("{Control>}a{/Control}");
    await userEvent.keyboard("{Delete}");
    expect(input).toHaveValue("");
  },
};
