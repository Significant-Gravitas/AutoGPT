import type { Meta, StoryObj } from "@storybook/react";
import { Button } from "./button";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "UI/Button",
  component: Button,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    variant: {
      control: "select",
      options: [
        "default",
        "destructive",
        "outline",
        "secondary",
        "ghost",
        "link",
      ],
    },
    size: {
      control: "select",
      options: ["default", "sm", "lg", "primary", "icon"],
    },
    disabled: {
      control: "boolean",
    },
    asChild: {
      control: "boolean",
    },
    children: {
      control: "text",
    },
    onClick: { action: "clicked" },
  },
} satisfies Meta<typeof Button>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: "Button",
  },
};

export const Interactive: Story = {
  args: {
    children: "Interactive Button",
  },
  argTypes: {
    onClick: { action: "clicked" },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: /Interactive Button/i });
    await userEvent.click(button);
    await expect(button).toHaveFocus();
  },
};

export const Variants: Story = {
  render: (args) => (
    <div className="flex flex-wrap gap-2">
      <Button {...args} variant="default">
        Default
      </Button>
      <Button {...args} variant="destructive">
        Destructive
      </Button>
      <Button {...args} variant="outline">
        Outline
      </Button>
      <Button {...args} variant="secondary">
        Secondary
      </Button>
      <Button {...args} variant="ghost">
        Ghost
      </Button>
      <Button {...args} variant="link">
        Link
      </Button>
    </div>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const buttons = canvas.getAllByRole("button");
    expect(buttons).toHaveLength(6);
    for (const button of buttons) {
      await userEvent.hover(button);
      await expect(button).toHaveAttribute(
        "class",
        expect.stringContaining("hover:"),
      );
    }
  },
};

export const Sizes: Story = {
  render: (args) => (
    <div className="flex flex-wrap items-center gap-2">
      <Button {...args} size="sm">
        Small
      </Button>
      <Button {...args} size="default">
        Default
      </Button>
      <Button {...args} size="lg">
        Large
      </Button>
      <Button {...args} size="primary">
        Primary
      </Button>
      <Button {...args} size="icon">
        🚀
      </Button>
    </div>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const buttons = canvas.getAllByRole("button");
    expect(buttons).toHaveLength(5);
    const sizes = ["sm", "default", "lg", "primary", "icon"];
    const sizeClasses = [
      "h-8 rounded-md px-3 text-xs",
      "h-9 px-4 py-2",
      "h-10 rounded-md px-8",
      "md:h-14 md:w-44 rounded-2xl h-10 w-28",
      "h-9 w-9",
    ];
    buttons.forEach((button, index) => {
      expect(button).toHaveAttribute(
        "class",
        expect.stringContaining(sizeClasses[index]),
      );
    });
  },
};

export const Disabled: Story = {
  args: {
    children: "Disabled Button",
    disabled: true,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: /Disabled Button/i });
    await expect(button).toBeDisabled();
    await expect(button).toHaveStyle("pointer-events: none");
    await expect(button).not.toHaveFocus();
  },
};

export const WithIcon: Story = {
  args: {
    children: (
      <>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="mr-2 h-4 w-4"
        >
          <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
        </svg>
        Button with Icon
      </>
    ),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: /Button with Icon/i });
    const icon = button.querySelector("svg");
    await expect(icon).toBeInTheDocument();
    await expect(button).toHaveTextContent("Button with Icon");
  },
};

export const LoadingState: Story = {
  args: {
    children: "Loading...",
    disabled: true,
  },
  render: (args) => (
    <Button {...args}>
      <svg
        className="mr-2 h-4 w-4 animate-spin"
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>
      {args.children}
    </Button>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: /Loading.../i });
    expect(button).toBeDisabled();
    const spinner = button.querySelector("svg");
    await expect(spinner).toHaveClass("animate-spin");
  },
};
