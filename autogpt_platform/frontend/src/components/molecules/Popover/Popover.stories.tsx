import { Button } from "@/components/atoms/Button/Button";
import { extendedButtonVariants } from "@/components/atoms/Button/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { ComponentProps, useState } from "react";
import { fn, userEvent, within } from "storybook/test";
import { Popover, PopoverContent, PopoverTrigger } from "./Popover";

const triggerClassName = extendedButtonVariants({
  variant: "secondary",
  size: "md",
});

const meta = {
  title: "Molecules/Popover",
  component: Popover,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          'Popover on Kobra (Base UI): `Popover` (root), `PopoverTrigger`, `PopoverContent` (portalled, 18rem wide, with arrow, `align="center"` and `sideOffset=8` by default) and `PopoverClose`. The content has `role="dialog"` but no built-in name, so pass `aria-label` or `aria-labelledby`. Opens with `defaultOpen` or a controlled `open`/`onOpenChange`.',
      },
      story: { height: "320px" },
    },
  },
  argTypes: {
    defaultOpen: {
      control: "boolean",
      description: "Open on first render (uncontrolled)",
    },
    modal: {
      control: "boolean",
      description: "Trap focus and block outside interaction while open",
    },
  },
  args: {
    defaultOpen: false,
    modal: false,
    onOpenChange: fn(),
  },
  render: renderBasic,
} satisfies Meta<typeof Popover>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const Open: Story = {
  args: { defaultOpen: true },
};

export const OpenedFromTrigger: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: "Run details" }),
    );
  },
};

export const Sides: Story = {
  render: renderSides,
  parameters: { layout: "padded" },
};

export const Alignments: Story = {
  render: renderAlignments,
  parameters: { layout: "padded" },
};

export const WithForm: Story = {
  render: renderWithForm,
};

export const Controlled: Story = {
  render: renderControlled,
};

type PopoverStoryProps = ComponentProps<typeof Popover>;

function renderBasic(args: PopoverStoryProps) {
  return (
    <Popover {...args}>
      <PopoverTrigger className={triggerClassName}>
        <Icon icon={InformationCircleIcon} size={16} />
        Run details
      </PopoverTrigger>
      <PopoverContent aria-label="Run details">
        <div className="flex flex-col gap-1">
          <Text variant="body-medium">Weekly report agent</Text>
          <Text variant="small" tone="secondary">
            Last run finished on 15 Jan 2026 at 10:30 and used 12 credits.
          </Text>
        </div>
      </PopoverContent>
    </Popover>
  );
}

function renderSides() {
  const sides = ["top", "right", "bottom", "left"] as const;

  return (
    <div className="grid grid-cols-2 gap-x-64 gap-y-40 px-48 py-24">
      {sides.map((side) => (
        <Popover key={side} defaultOpen>
          <PopoverTrigger className={triggerClassName}>{side}</PopoverTrigger>
          <PopoverContent
            side={side}
            className="w-40"
            aria-label={`Popover on ${side}`}
          >
            <Text variant="small">Opens on the {side} side.</Text>
          </PopoverContent>
        </Popover>
      ))}
    </div>
  );
}

function renderAlignments() {
  const alignments = ["start", "center", "end"] as const;

  return (
    <div className="flex flex-col items-center gap-32 pb-24">
      {alignments.map((align) => (
        <Popover key={align} defaultOpen>
          <PopoverTrigger className={triggerClassName}>
            align {align}
          </PopoverTrigger>
          <PopoverContent align={align} aria-label={`Aligned to ${align}`}>
            <Text variant="small">
              Content aligned to the {align} of its trigger.
            </Text>
          </PopoverContent>
        </Popover>
      ))}
    </div>
  );
}

function renderWithForm() {
  return (
    <Popover defaultOpen>
      <PopoverTrigger className={triggerClassName}>Rename agent</PopoverTrigger>
      <PopoverContent aria-labelledby="rename-agent-title" className="w-80">
        <div className="flex flex-col gap-3">
          <Text variant="body-medium" id="rename-agent-title">
            Rename agent
          </Text>
          <Input
            id="agent-name"
            label="Agent name"
            defaultValue="Weekly report agent"
            wrapperClassName="mb-0!"
          />
          <div className="flex justify-end gap-2">
            <Button variant="secondary" size="md">
              Cancel
            </Button>
            <Button variant="primary" size="md">
              Save
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

function ControlledPopover() {
  const [open, setOpen] = useState(false);

  return (
    <div className="flex flex-col items-center gap-3">
      <Text variant="small" tone="muted">
        Popover is {open ? "open" : "closed"}
      </Text>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger className={triggerClassName}>
          Share agent
        </PopoverTrigger>
        <PopoverContent aria-label="Share agent">
          <div className="flex flex-col gap-3">
            <Text variant="body">Anyone with the link can run this agent.</Text>
            <Button variant="primary" size="md" onClick={() => setOpen(false)}>
              Done
            </Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}

function renderControlled() {
  return <ControlledPopover />;
}
