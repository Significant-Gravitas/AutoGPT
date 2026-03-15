import type { Meta, StoryObj } from "@storybook/nextjs";
import { Button } from "../Button/Button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from "./BaseTooltip";

const meta: Meta<typeof Tooltip> = {
  title: "Atoms/Tooltip",
  tags: ["autodocs"],
  component: Tooltip,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Tooltip component built on Radix UI primitives. Provides contextual information on hover with customizable delay and positioning. Includes TooltipProvider, Tooltip, TooltipTrigger, and TooltipContent components.",
      },
    },
  },
  argTypes: {
    delayDuration: {
      control: { type: "number", min: 0, max: 2000, step: 100 },
      description: "Delay in milliseconds before tooltip appears",
    },
    children: {
      control: false,
      description: "Tooltip content and trigger elements",
    },
  },
  args: {
    delayDuration: 10,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: function DefaultTooltip(args) {
    return (
      <TooltipProvider>
        <Tooltip delayDuration={args.delayDuration}>
          <TooltipTrigger asChild>
            <Button variant="secondary">Hover me</Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>This is a tooltip</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  },
};

export const WithDelay: Story = {
  render: function DelayedTooltip() {
    return (
      <TooltipProvider>
        <Tooltip delayDuration={1000}>
          <TooltipTrigger asChild>
            <Button variant="secondary">Hover me (1s delay)</Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>This tooltip appears after 1 second</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  },
  parameters: {
    docs: {
      description: {
        story:
          "Tooltip with a longer delay duration to demonstrate the timing control.",
      },
    },
  },
};

export const LongContent: Story = {
  render: function LongContentTooltip() {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="secondary">Long content</Button>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            <p>
              This is a tooltip with longer content that demonstrates how the
              tooltip handles text wrapping and maintains readability with
              extended descriptions.
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  },
};

export const DifferentSides: Story = {
  render: function DifferentSidesTooltip() {
    return (
      <div className="flex items-center gap-8">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="secondary" size="small">
                Top
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p>Tooltip on top</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="secondary" size="small">
                Right
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>Tooltip on right</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="secondary" size="small">
                Bottom
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>Tooltip on bottom</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="secondary" size="small">
                Left
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">
              <p>Tooltip on left</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    );
  },
  parameters: {
    docs: {
      description: {
        story:
          "Tooltips can be positioned on different sides of the trigger element.",
      },
    },
  },
};

export const WithIcon: Story = {
  render: function IconTooltip() {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <button className="rounded-full p-2 hover:bg-gray-100 dark:hover:bg-gray-800">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M9,9h0a3,3,0,0,1,6,0c0,2-3,3-3,3" />
                <path d="m12,17h0" />
              </svg>
            </button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Help information</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  },
  parameters: {
    docs: {
      description: {
        story: "Tooltip can be used with icon buttons for help or information.",
      },
    },
  },
};

export const MultipleTooltips: Story = {
  render: function MultipleTooltips() {
    return (
      <TooltipProvider>
        <div className="flex items-center gap-4">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="secondary" size="small">
                Save
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Save your changes</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="secondary" size="small">
                Edit
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Edit this item</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="destructive" size="small">
                Delete
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Delete this item permanently</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>
    );
  },
  parameters: {
    docs: {
      description: {
        story:
          "Multiple tooltips can share a single TooltipProvider for better performance.",
      },
    },
  },
};
