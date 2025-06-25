import type { Meta, StoryObj } from "@storybook/nextjs";
import { Play, Plus } from "lucide-react";
import { expect, fn, userEvent, within } from "storybook/test";
import { Button } from "./Button";

const meta: Meta<typeof Button> = {
  title: "Atoms/Tests/Button",
  component: Button,
  parameters: {
    layout: "centered",
  },
  tags: ["!autodocs"],
  args: {
    onClick: fn(),
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Test button click functionality
export const ClickInteraction: Story = {
  args: {
    children: "Click Me",
    variant: "primary",
  },
  play: async function testButtonClick({ args, canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Click Me" });

    // Test initial state
    expect(button).toBeInTheDocument();
    expect(button).not.toBeDisabled();

    // Test click interaction
    await userEvent.click(button);

    // Assert the click handler was called
    expect(args.onClick).toHaveBeenCalledTimes(1);
  },
};

// Test disabled button behavior
export const DisabledInteraction: Story = {
  args: {
    children: "Disabled Button",
    variant: "primary",
    disabled: true,
  },
  play: async function testDisabledButton({ canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Disabled Button" });

    // Test disabled state
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("disabled");

    // Test that disabled button has proper styling (pointer-events: none prevents clicking)
    // We don't test clicking because disabled buttons with pointer-events: none can't be clicked
  },
};

// Test loading button behavior
export const LoadingInteraction: Story = {
  args: {
    children: "Loading Button",
    variant: "primary",
    loading: true,
  },
  play: async function testLoadingButton({ canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Loading Button" });

    // Test loading state - button should show loading spinner
    const spinner = button.querySelector("svg");
    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass("animate-spin");

    // Test that loading button is still clickable but pointer events are disabled
    expect(button).toHaveClass("pointer-events-none");
  },
};

// Test keyboard navigation
export const KeyboardInteraction: Story = {
  args: {
    children: "Keyboard Test",
    variant: "primary",
  },
  play: async function testKeyboardNavigation({ args, canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Keyboard Test" });

    // Test tab navigation
    await userEvent.tab();
    expect(button).toHaveFocus();

    // Test Enter key activation
    await userEvent.keyboard("{Enter}");
    expect(args.onClick).toHaveBeenCalledTimes(1);

    // Test Space key activation
    await userEvent.keyboard(" ");
    expect(args.onClick).toHaveBeenCalledTimes(2);
  },
};

// Test focus and blur events
export const FocusInteraction: Story = {
  args: {
    children: "Focus Test",
    variant: "outline",
  },
  play: async function testFocusEvents({ canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Focus Test" });

    // Test programmatic focus
    button.focus();
    expect(button).toHaveFocus();

    // Test blur
    button.blur();
    expect(button).not.toHaveFocus();

    // Test click focus
    await userEvent.click(button);
    expect(button).toHaveFocus();
  },
};

// Test different variants work correctly
export const VariantsInteraction: Story = {
  render: function renderVariants(args) {
    return (
      <div className="flex gap-4">
        <Button variant="primary" onClick={args.onClick}>
          Primary
        </Button>
        <Button variant="secondary" onClick={args.onClick}>
          Secondary
        </Button>
        <Button variant="destructive" onClick={args.onClick}>
          Destructive
        </Button>
        <Button variant="outline" onClick={args.onClick}>
          Outline
        </Button>
        <Button variant="ghost" onClick={args.onClick}>
          Ghost
        </Button>
      </div>
    );
  },
  play: async function testVariants({ args, canvasElement }) {
    const canvas = within(canvasElement);

    // Test all variants are rendered
    const primaryBtn = canvas.getByRole("button", { name: "Primary" });
    const secondaryBtn = canvas.getByRole("button", { name: "Secondary" });
    const destructiveBtn = canvas.getByRole("button", { name: "Destructive" });
    const outlineBtn = canvas.getByRole("button", { name: "Outline" });
    const ghostBtn = canvas.getByRole("button", { name: "Ghost" });

    expect(primaryBtn).toBeInTheDocument();
    expect(secondaryBtn).toBeInTheDocument();
    expect(destructiveBtn).toBeInTheDocument();
    expect(outlineBtn).toBeInTheDocument();
    expect(ghostBtn).toBeInTheDocument();

    // Test clicking each variant
    await userEvent.click(primaryBtn);
    await userEvent.click(secondaryBtn);
    await userEvent.click(destructiveBtn);
    await userEvent.click(outlineBtn);
    await userEvent.click(ghostBtn);

    // Assert all clicks were registered
    expect(args.onClick).toHaveBeenCalledTimes(5);
  },
};

// Test button with icons
export const IconInteraction: Story = {
  args: {
    children: "With Icon",
    variant: "primary",
    leftIcon: <Play className="h-4 w-4" />,
    rightIcon: <Plus className="h-4 w-4" />,
  },
  play: async function testIconButton({ args, canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "With Icon" });

    // Test button with icons is rendered correctly
    expect(button).toBeInTheDocument();

    // Test icons are present
    const icons = button.querySelectorAll("svg");
    expect(icons).toHaveLength(2); // leftIcon + rightIcon

    // Test button functionality with icons
    await userEvent.click(button);
    expect(args.onClick).toHaveBeenCalledTimes(1);
  },
};

// Test icon-only button
export const IconOnlyInteraction: Story = {
  args: {
    children: <Plus className="h-4 w-4" />,
    variant: "icon",
    size: "icon",
    "aria-label": "Add item",
  },
  play: async function testIconOnlyButton({ args, canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Add item" });

    // Test icon-only button accessibility
    expect(button).toHaveAccessibleName("Add item");

    // Test icon is present
    const icon = button.querySelector("svg");
    expect(icon).toBeInTheDocument();

    // Test functionality
    await userEvent.click(button);
    expect(args.onClick).toHaveBeenCalledTimes(1);
  },
};

// Test multiple clicks and double click
export const MultipleClicksInteraction: Story = {
  args: {
    children: "Multi Click",
    variant: "secondary",
  },
  play: async function testMultipleClicks({ args, canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Multi Click" });

    // Test multiple single clicks
    await userEvent.click(button);
    await userEvent.click(button);
    await userEvent.click(button);

    expect(args.onClick).toHaveBeenCalledTimes(3);

    // Test double click
    await userEvent.dblClick(button);
    expect(args.onClick).toHaveBeenCalledTimes(5); // 3 + 2 from double click
  },
};

// Test hover states
export const HoverInteraction: Story = {
  args: {
    children: "Hover Me",
    variant: "outline",
  },
  play: async function testHoverStates({ canvasElement }) {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: "Hover Me" });

    // Test hover interaction
    await userEvent.hover(button);

    // Test unhover
    await userEvent.unhover(button);

    // Verify button is still functional after hover interactions
    expect(button).toBeInTheDocument();
  },
};
