import type { Meta, StoryObj } from "@storybook/nextjs";
import { Collapsible } from "./Collapsible";

const meta: Meta<typeof Collapsible> = {
  title: "Molecules/Collapsible",
  component: Collapsible,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: `
## Collapsible Component

A reusable collapsible component built on top of shadcn's collapsible primitives with enhanced functionality and styling.

### ✨ Features

- **Built on shadcn base** - Uses shadcn collapsible primitives without modification
- **Custom trigger design** - Enhanced trigger with "↓ more" / "↑ less" text indicators
- **Smooth animations** - Chevron rotation and content expand/collapse transitions
- **Controlled & uncontrolled modes** - Supports both controlled and uncontrolled usage
- **Customizable styling** - Props for custom classes on trigger, content, and root
- **Accessible** - Built on Radix UI primitives for full accessibility support
- **TypeScript support** - Complete TypeScript interface support

### 🎯 Usage

\`\`\`tsx
<Collapsible 
  trigger={<span>Click to expand</span>}
  defaultOpen={false}
>
  <p>This content will be collapsed/expanded</p>
</Collapsible>
\`\`\`

### Props

- **trigger**: React node to display as the clickable trigger
- **children**: Content to show/hide when collapsing/expanding
- **defaultOpen**: Initial open state (uncontrolled mode)
- **open**: Current open state (controlled mode)
- **onOpenChange**: Callback when open state changes
- **className**: Additional classes for the root container
- **triggerClassName**: Additional classes for the trigger element
- **contentClassName**: Additional classes for the content container
        `,
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    trigger: {
      control: false,
      description: "The trigger element to click for expanding/collapsing",
    },
    children: {
      control: false,
      description: "Content to show when expanded",
    },
    defaultOpen: {
      control: "boolean",
      description: "Initial open state (uncontrolled mode)",
      table: {
        defaultValue: { summary: "false" },
      },
    },
    open: {
      control: "boolean",
      description: "Current open state (controlled mode)",
    },
    onOpenChange: {
      control: false,
      description: "Callback function when open state changes",
    },
    className: {
      control: "text",
      description: "Additional CSS classes for root container",
    },
    triggerClassName: {
      control: "text",
      description: "Additional CSS classes for trigger element",
    },
    contentClassName: {
      control: "text",
      description: "Additional CSS classes for content container",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Default collapsible with simple text trigger and content.
 */
export const Default: Story = {
  args: {
    trigger: <span className="font-medium">Click to expand</span>,
    children: (
      <div className="space-y-2">
        <p>
          This is the collapsible content that can be expanded or collapsed.
        </p>
        <p>You can put any React elements here.</p>
      </div>
    ),
    defaultOpen: false,
  },
};

/**
 * Collapsible that starts in the expanded state.
 */
export const DefaultOpen: Story = {
  args: {
    trigger: <span className="font-medium">Already expanded</span>,
    children: (
      <div className="space-y-2">
        <p>This collapsible starts in the open state.</p>
        <p>Notice how the chevron and text indicators reflect this.</p>
      </div>
    ),
    defaultOpen: true,
  },
};

/**
 * Multiple collapsibles can be used together to create accordion-like interfaces.
 */
export const MultipleCollapsibles: Story = {
  render: () => (
    <div className="w-96 space-y-4">
      <Collapsible
        trigger={<span className="font-medium">Section 1</span>}
        className="p-3"
      >
        <p>Content for the first section.</p>
      </Collapsible>
      <Collapsible
        trigger={<span className="font-medium">Section 2</span>}
        className="p-3"
        defaultOpen={true}
      >
        <p>Content for the second section, which starts expanded.</p>
      </Collapsible>
      <Collapsible
        trigger={<span className="font-medium">Section 3</span>}
        className="p-3"
      >
        <p>Content for the third section.</p>
      </Collapsible>
    </div>
  ),
};
