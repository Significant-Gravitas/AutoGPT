import type { Meta, StoryObj } from "@storybook/react";
import { Button } from "@/components/ui/button";
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
      options: ["default", "sm", "lg", "icon"],
      description: "Button size variants. The 'primary' size is deprecated.",
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
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole("button", { name: /Button/i });

    // Test default styling
    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("rounded-full"),
    );

    // Test SVG styling is present
    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("[&_svg]:size-4"),
    );

    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("[&_svg]:shrink-0"),
    );

    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("[&_svg]:pointer-events-none"),
    );
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

    // Test interaction
    await userEvent.click(button);
    await expect(button).toHaveFocus();

    // Test styling matches the updated component
    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("rounded-full"),
    );

    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("gap-2"),
    );

    // Test other key button styles
    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("inline-flex items-center justify-center"),
    );
  },
};

export const Variants: Story = {
  render: (args) => (
    <div className="flex flex-wrap gap-4">
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
    await expect(buttons).toHaveLength(6);

    // Test hover states
    for (const button of buttons) {
      await userEvent.hover(button);
      await expect(button).toHaveAttribute(
        "class",
        expect.stringContaining("hover:"),
      );
    }

    // Test rounded-full styling on appropriate variants
    const roundedVariants = [
      "default",
      "destructive",
      "outline",
      "secondary",
      "ghost",
    ];
    for (let i = 0; i < 5; i++) {
      await expect(buttons[i]).toHaveAttribute(
        "class",
        expect.stringContaining("rounded-full"),
      );
    }

    // Link variant should not have rounded-full
    await expect(buttons[5]).not.toHaveAttribute(
      "class",
      expect.stringContaining("rounded-full"),
    );
  },
};

export const Sizes: Story = {
  render: (args) => (
    <div className="flex flex-wrap items-center gap-4">
      <Button {...args} size="icon">
        🚀
      </Button>
      <Button {...args} size="sm">
        Small
      </Button>
      <Button {...args}>Default</Button>
      <Button {...args} size="lg">
        Large
      </Button>
      <div className="flex flex-col items-start gap-2 rounded border p-4">
        <p className="mb-2 text-xs text-muted-foreground">Deprecated Size:</p>
        <Button {...args} size="primary">
          Primary (deprecated)
        </Button>
      </div>
    </div>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const buttons = canvas.getAllByRole("button");
    await expect(buttons).toHaveLength(5);

    // Test icon size
    const iconButton = canvas.getByRole("button", { name: /🚀/i });
    await expect(iconButton).toHaveAttribute(
      "class",
      expect.stringContaining("h-9 w-9"),
    );

    // Test specific size classes
    const smallButton = canvas.getByRole("button", { name: /Small/i });
    await expect(smallButton).toHaveAttribute(
      "class",
      expect.stringContaining("h-8"),
    );

    const defaultButton = canvas.getByRole("button", { name: /Default/i });
    await expect(defaultButton).toHaveAttribute(
      "class",
      expect.stringContaining("h-9"),
    );

    const largeButton = canvas.getByRole("button", { name: /Large/i });
    await expect(largeButton).toHaveAttribute(
      "class",
      expect.stringContaining("h-10"),
    );
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
    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("disabled:pointer-events-none"),
    );
    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("disabled:opacity-50"),
    );
    await expect(button).not.toHaveFocus();
  },
};

export const WithIcon: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex gap-4">
        <Button>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
          </svg>
          Icon Left
        </Button>
        <Button>
          Icon Right
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
          </svg>
        </Button>
      </div>
      <div>
        <p className="mb-2 text-sm text-muted-foreground">
          Icon with automatic gap spacing:
        </p>
        <Button>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
          </svg>
          Button with Icon
        </Button>
      </div>
    </div>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const buttons = canvas.getAllByRole("button");
    const icons = canvasElement.querySelectorAll("svg");

    // Test that SVGs are present
    await expect(icons.length).toBeGreaterThan(0);

    // Test for gap-2 class for spacing
    await expect(buttons[0]).toHaveAttribute(
      "class",
      expect.stringContaining("gap-2"),
    );

    // Test SVG styling from buttonVariants
    await expect(buttons[0]).toHaveAttribute(
      "class",
      expect.stringContaining("[&_svg]:size-4"),
    );

    await expect(buttons[0]).toHaveAttribute(
      "class",
      expect.stringContaining("[&_svg]:shrink-0"),
    );

    await expect(buttons[0]).toHaveAttribute(
      "class",
      expect.stringContaining("[&_svg]:pointer-events-none"),
    );
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
        className="animate-spin"
        xmlns="http://www.w3.org/2000/svg"
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
    await expect(button).toBeDisabled();
    const spinner = button.querySelector("svg");
    await expect(spinner).toHaveClass("animate-spin");

    // Test SVG styling from buttonVariants
    await expect(button).toHaveAttribute(
      "class",
      expect.stringContaining("[&_svg]:size-4"),
    );
  },
};

export const RoundedStyles: Story = {
  render: () => (
    <div className="flex flex-col gap-6">
      <div>
        <p className="mb-2 text-sm text-muted-foreground">
          Default variants have rounded-full style:
        </p>
        <div className="flex gap-4">
          <Button variant="default">Default</Button>
          <Button variant="destructive">Destructive</Button>
          <Button variant="outline">Outline</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="ghost">Ghost</Button>
        </div>
      </div>
      <div>
        <p className="mb-2 text-sm text-muted-foreground">
          Link variant maintains its original style:
        </p>
        <div className="flex gap-4">
          <Button variant="link">Link</Button>
        </div>
      </div>
    </div>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const buttons = canvas.getAllByRole("button");

    // Test rounded-full on first 5 buttons
    for (let i = 0; i < 5; i++) {
      await expect(buttons[i]).toHaveAttribute(
        "class",
        expect.stringContaining("rounded-full"),
      );
    }

    // Test that link variant doesn't have rounded-full
    await expect(buttons[5]).not.toHaveAttribute(
      "class",
      expect.stringContaining("rounded-full"),
    );
  },
};
