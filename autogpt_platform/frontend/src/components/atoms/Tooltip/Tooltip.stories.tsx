import type { Meta, StoryObj } from "@storybook/nextjs-vite";
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
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    layout: "centered",
    docs: {
      description: {
        component:
          "Tooltip on Kobra (Base UI). Provides contextual information on hover with customizable delay and positioning. Includes TooltipProvider, Tooltip, TooltipTrigger (takes the trigger element through `render`) and TooltipContent.",
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
          <TooltipTrigger
            render={<Button variant="secondary">Hover me</Button>}
          />
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
          <TooltipTrigger
            render={<Button variant="secondary">Hover me (1s delay)</Button>}
          />
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
          <TooltipTrigger
            render={<Button variant="secondary">Long content</Button>}
          />
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
            <TooltipTrigger
              render={
                <Button variant="secondary" size="md">
                  Top
                </Button>
              }
            />
            <TooltipContent side="top">
              <p>Tooltip on top</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button variant="secondary" size="md">
                  Right
                </Button>
              }
            />
            <TooltipContent side="right">
              <p>Tooltip on right</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button variant="secondary" size="md">
                  Bottom
                </Button>
              }
            />
            <TooltipContent side="bottom">
              <p>Tooltip on bottom</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button variant="secondary" size="md">
                  Left
                </Button>
              }
            />
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
          <TooltipTrigger
            render={
              <button
                className="rounded-full p-2 hover:bg-zinc-100"
                aria-label="More information"
              >
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
            }
          />
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
            <TooltipTrigger
              render={
                <Button variant="secondary" size="md">
                  Save
                </Button>
              }
            />
            <TooltipContent>
              <p>Save your changes</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger
              render={
                <Button variant="secondary" size="md">
                  Edit
                </Button>
              }
            />
            <TooltipContent>
              <p>Edit this item</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger
              render={
                <Button variant="destructive" size="md">
                  Delete
                </Button>
              }
            />
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
