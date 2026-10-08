import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  Cancel01Icon,
  FilterHorizontalIcon,
  GridViewIcon,
  ListViewIcon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  PlayIcon,
  PlusSignIcon,
  SparklesIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { TooltipProvider } from "../Tooltip/BaseTooltip";
import { Button } from "./Button";

const meta: Meta<typeof Button> = {
  title: "Atoms/Button",
  tags: ["autodocs"],
  component: Button,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
  parameters: {
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    layout: "centered",
    docs: {
      description: {
        component:
          "Button component with multiple variants and sizes based on our design system. Built on Kobra's Button (Base UI): house variants map onto Kobra's, pills via `rounded`, loading through Kobra's Spinner with `aria-busy`.",
      },
    },
  },
  argTypes: {
    variant: {
      control: "select",
      options: [
        "primary",
        "secondary",
        "destructive",
        "outline",
        "ghost",
        "icon",
        "floating",
        "toggle",
        "link",
        "loading",
      ],
      description: "Button style variant",
    },
    size: {
      control: "select",
      options: ["sm", "md", "lg", "icon-sm", "icon-md", "icon-lg"],
      description: "Button size",
    },
    loading: {
      control: "boolean",
      description: "Show loading spinner and disable button",
    },
    disabled: {
      control: "boolean",
      description: "Disable the button",
    },
    children: {
      control: "text",
      description: "Button content",
    },
  },
  args: {
    children: "Button",
    variant: "primary",
    size: "lg",
    loading: false,
    disabled: false,
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

// Basic variants
export const Primary: Story = {
  args: {
    variant: "primary",
    children: "Primary Button",
  },
};

export const Secondary: Story = {
  args: {
    variant: "secondary",
    children: "Secondary Button",
  },
};

export const Destructive: Story = {
  args: {
    variant: "destructive",
    children: "Delete",
  },
};

export const Outline: Story = {
  args: {
    variant: "outline",
    children: "Outline Button",
  },
};

export const Ghost: Story = {
  args: {
    variant: "ghost",
    children: "Ghost Button",
  },
};

export const LinkVariant: Story = {
  args: {
    variant: "link",
    children: "Go to documentation",
  },
};

// Loading states
export const Loading: Story = {
  args: {
    variant: "primary",
    loading: true,
    children: "Saving...",
  },
  parameters: {
    docs: {
      description: {
        story:
          "Use contextual loading text that reflects the action being performed (e.g., 'Computing...', 'Processing...', 'Saving...', 'Uploading...', 'Deleting...')",
      },
    },
  },
};

export const LoadingLink: Story = {
  args: {
    variant: "link",
    loading: true,
    children: "Loading link",
  },
  parameters: {
    docs: {
      description: {
        story:
          "Link buttons inherit the secondary link styling while respecting the loading state.",
      },
    },
  },
};

export const LoadingGhost: Story = {
  args: {
    variant: "ghost",
    loading: true,
    children: "Fetching data...",
  },
  parameters: {
    docs: {
      description: {
        story:
          "Always show contextual loading text that describes what's happening. Avoid generic 'Loading...' text when possible.",
      },
    },
  },
};

// Contextual loading examples
export const ContextualLoadingExamples: Story = {
  render: renderContextualLoadingExamples,
  parameters: {
    docs: {
      description: {
        story:
          "Examples of contextual loading text. Always use specific action-based text rather than generic 'Loading...' to give users clear feedback about what's happening.",
      },
    },
  },
};

// Sizes
export const SmallButtons: Story = {
  render: renderSmallButtons,
};

export const LargeButtons: Story = {
  render: renderLargeButtons,
};

// Compact actions (team, home)
export const ActionButtons: Story = {
  render: renderActionButtons,
  parameters: {
    docs: {
      description: {
        story:
          '`size="sm"` is the 32px rounded-rectangle chip used for row and header actions across Home and Team. Pair with `leadingIcon` for a 14px Hugeicon.',
      },
    },
  },
};

export const IconButtons: Story = {
  render: renderIconButtons,
  parameters: {
    docs: {
      description: {
        story:
          '`size="icon-sm"` (32px), `size="icon-md"` (36px) and `size="icon-lg"` (40px) are square icon-only actions. `aria-label` is required and doubles as the tooltip. `variant="floating"` sits over images and card covers.',
      },
    },
  },
};

export const ToggleButtons: Story = {
  render: renderToggleButtons,
  parameters: {
    docs: {
      description: {
        story:
          '`variant="toggle"` reads `aria-pressed` for its on state. Use it for filter chips and segmented controls.',
      },
    },
  },
};

// With icons
export const WithLeftIcon: Story = {
  args: {
    variant: "primary",
    leftIcon: <Icon icon={PlayIcon} size={16} />,
    children: "Play",
  },
};

export const WithRightIcon: Story = {
  args: {
    variant: "outline",
    rightIcon: <Icon icon={PlusSignIcon} size={16} />,
    children: "Add Item",
  },
};

export const IconOnly: Story = {
  args: {
    variant: "icon",
    size: "icon-lg",
    children: <Icon icon={PlusSignIcon} size={16} />,
    "aria-label": "Add item",
  },
};

// States
export const Disabled: Story = {
  render: renderDisabledButtons,
};

// Complete showcase matching Figma design
export const AllVariants: Story = {
  render: renderAllVariants,
};

// Render functions as function declarations
function renderActionButtons() {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm">
          Chat
        </Button>
        <Button variant="secondary" size="sm">
          Manage
        </Button>
        <Button variant="outline" size="sm">
          New Pod
        </Button>
        <Button variant="ghost" size="sm">
          Open in library
        </Button>
        <Button variant="destructive" size="sm">
          Fire Maria
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm" leadingIcon={SparklesIcon}>
          New Expert
        </Button>
        <Button variant="secondary" size="sm" leadingIcon={PencilEdit02Icon}>
          Edit Soul
        </Button>
        <Button variant="secondary" size="sm" leadingIcon={PlusSignIcon}>
          Install workflow
        </Button>
        <Button
          variant="ghost"
          size="sm"
          leadingIcon={FilterHorizontalIcon}
          className="text-zinc-600"
        >
          All work
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="secondary" size="sm" loading>
          Resuming...
        </Button>
        <Button variant="secondary" size="sm" disabled>
          Disabled
        </Button>
        <Button as="NextLink" href="#" variant="secondary" size="sm">
          Review
        </Button>
      </div>
    </div>
  );
}

function renderIconButtons() {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="icon"
          size="icon-sm"
          leadingIcon={FilterHorizontalIcon}
          aria-label="Filter workflows"
        />
        <Button
          variant="primary"
          size="icon-sm"
          leadingIcon={Tick02Icon}
          aria-label="Approve"
        />
        <Button
          variant="destructive"
          size="icon-sm"
          leadingIcon={Cancel01Icon}
          aria-label="Confirm decline"
        />
        <Button
          variant="ghost"
          size="icon-sm"
          leadingIcon={MoreHorizontalIcon}
          aria-label="More actions"
        />
      </div>
      <div className="flex flex-wrap items-center gap-2 rounded-lg bg-linear-to-r from-purple-200 to-sky-200 p-4">
        <Button
          variant="floating"
          size="icon-sm"
          leadingIcon={PencilEdit02Icon}
          aria-label="Edit workflow"
        />
        <Button
          variant="floating"
          size="icon-sm"
          leadingIcon={SparklesIcon}
          aria-label="Ask about this workflow"
        />
        <Button
          variant="floating"
          size="icon-sm"
          leadingIcon={PencilEdit02Icon}
          aria-label="Edit Soul"
        />
      </div>
    </div>
  );
}

function renderToggleButtons() {
  return (
    <div className="flex flex-wrap items-center gap-4">
      <Button variant="toggle" size="sm" aria-pressed={false}>
        Needs review (3)
      </Button>
      <Button
        variant="toggle"
        size="sm"
        aria-pressed
        className="border-zinc-200 bg-white aria-pressed:border-yellow-200 aria-pressed:bg-yellow-100 aria-pressed:text-yellow-700"
      >
        Needs review (3)
      </Button>
      <div className="flex h-7 items-center rounded-md border border-zinc-200 p-0.5">
        <Button
          variant="toggle"
          size="icon-sm"
          className="size-6 rounded-sm"
          leadingIcon={ListViewIcon}
          aria-label="List view"
          aria-pressed
        />
        <Button
          variant="toggle"
          size="icon-sm"
          className="size-6 rounded-sm"
          leadingIcon={GridViewIcon}
          aria-label="Grid view"
          aria-pressed={false}
        />
      </div>
    </div>
  );
}

function renderContextualLoadingExamples() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-4 text-base font-medium text-zinc-900">
          ✅ Good Examples - Contextual Loading Text
        </h3>
        <div className="flex flex-wrap gap-4">
          <Button variant="primary" loading>
            Saving...
          </Button>
          <Button variant="primary" loading>
            Computing...
          </Button>
          <Button variant="primary" loading>
            Processing...
          </Button>
          <Button variant="primary" loading>
            Uploading...
          </Button>
          <Button variant="destructive" loading>
            Deleting...
          </Button>
          <Button variant="secondary" loading>
            Generating...
          </Button>
          <Button variant="ghost" loading>
            Fetching data...
          </Button>
          <Button variant="outline" loading>
            Analyzing...
          </Button>
        </div>
      </div>

      <div>
        <h3 className="mb-4 text-base font-medium text-red-600">
          ❌ Avoid - Generic Loading Text
        </h3>
        <div className="flex flex-wrap gap-4">
          <Button variant="primary" loading disabled>
            Loading...
          </Button>
          <Button variant="secondary" loading disabled>
            Please wait...
          </Button>
          <Button variant="outline" loading disabled>
            Working...
          </Button>
        </div>
        <p className="mt-2 text-sm text-zinc-600">
          These examples are disabled to show what NOT to do. Use specific
          action-based text instead.
        </p>
      </div>
    </div>
  );
}

function renderSmallButtons() {
  return (
    <div className="flex flex-wrap gap-4">
      <Button variant="primary" size="md">
        Primary
      </Button>
      <Button variant="secondary" size="md">
        Secondary
      </Button>
      <Button variant="destructive" size="md">
        Delete
      </Button>
      <Button variant="outline" size="md">
        Outline
      </Button>
      <Button variant="ghost" size="md">
        Ghost
      </Button>
    </div>
  );
}

function renderLargeButtons() {
  return (
    <div className="flex flex-wrap gap-4">
      <Button variant="primary" size="lg">
        Primary
      </Button>
      <Button variant="secondary" size="lg">
        Secondary
      </Button>
      <Button variant="destructive" size="lg">
        Delete
      </Button>
      <Button variant="outline" size="lg">
        Outline
      </Button>
      <Button variant="ghost" size="lg">
        Ghost
      </Button>
    </div>
  );
}

function renderDisabledButtons() {
  return (
    <div className="flex flex-wrap gap-4">
      <Button variant="primary" disabled>
        Primary Disabled
      </Button>
      <Button variant="secondary" disabled>
        Secondary Disabled
      </Button>
      <Button variant="destructive" disabled>
        Destructive Disabled
      </Button>
      <Button variant="outline" disabled>
        Outline Disabled
      </Button>
      <Button variant="ghost" disabled>
        Ghost Disabled
      </Button>
    </div>
  );
}

function renderAllVariants() {
  return (
    <div className="space-y-12 p-8">
      {/* Large buttons section */}
      <div className="space-y-8">
        <h2 className="text-3xl font-semibold text-zinc-900">Large buttons</h2>
        <div className="flex flex-wrap gap-20">
          {/* Primary */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Primary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="primary" size="lg">
                Save
              </Button>
              <Button variant="primary" size="lg" loading>
                Loading
              </Button>
              <Button variant="primary" size="lg" disabled>
                Disabled
              </Button>
              <Button
                variant="primary"
                size="lg"
                leftIcon={<Icon icon={PlayIcon} size={20} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Secondary */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Secondary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="secondary" size="lg">
                Save
              </Button>
              <Button variant="secondary" size="lg" loading>
                Loading
              </Button>
              <Button variant="secondary" size="lg" disabled>
                Disabled
              </Button>
              <Button
                variant="secondary"
                size="lg"
                leftIcon={<Icon icon={PlayIcon} size={20} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Destructive */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Destructive
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="destructive" size="lg">
                Save
              </Button>
              <Button variant="destructive" size="lg" loading>
                Loading
              </Button>
              <Button variant="destructive" size="lg" disabled>
                Disabled
              </Button>
              <Button
                variant="destructive"
                size="lg"
                leftIcon={<Icon icon={PlayIcon} size={20} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Outline */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Outline
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="outline" size="lg">
                Save
              </Button>
              <Button variant="outline" size="lg" loading>
                Loading
              </Button>
              <Button variant="outline" size="lg" disabled>
                Disabled
              </Button>
              <Button
                variant="outline"
                size="lg"
                leftIcon={<Icon icon={PlayIcon} size={20} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Ghost */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Save
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="ghost" size="lg">
                Text
              </Button>
              <Button variant="ghost" size="lg" loading>
                Loading
              </Button>
              <Button variant="ghost" size="lg" disabled>
                Disabled
              </Button>
              <Button
                variant="ghost"
                size="lg"
                leftIcon={<Icon icon={PlayIcon} size={20} />}
              >
                Play
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Small buttons section */}
      <div className="space-y-8">
        <h2 className="text-3xl font-semibold text-zinc-900">Small buttons</h2>
        <div className="flex flex-wrap gap-20">
          {/* Primary Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Primary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="primary" size="md">
                Save
              </Button>
              <Button variant="primary" size="md" loading>
                Loading
              </Button>
              <Button variant="primary" size="md" disabled>
                Disabled
              </Button>
              <Button
                variant="primary"
                size="md"
                leftIcon={<Icon icon={PlayIcon} size={16} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Secondary Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Secondary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="secondary" size="md">
                Save
              </Button>
              <Button variant="secondary" size="md" loading>
                Loading
              </Button>
              <Button variant="secondary" size="md" disabled>
                Disabled
              </Button>
              <Button
                variant="secondary"
                size="md"
                leftIcon={<Icon icon={PlayIcon} size={16} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Destructive Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Destructive
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="destructive" size="md">
                Save
              </Button>
              <Button variant="destructive" size="md" loading>
                Loading
              </Button>
              <Button variant="destructive" size="md" disabled>
                Disabled
              </Button>
              <Button
                variant="destructive"
                size="md"
                leftIcon={<Icon icon={PlayIcon} size={16} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Outline Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Outline
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="outline" size="md">
                Save
              </Button>
              <Button variant="outline" size="md" loading>
                Loading
              </Button>
              <Button variant="outline" size="md" disabled>
                Disabled
              </Button>
              <Button
                variant="outline"
                size="md"
                leftIcon={<Icon icon={PlayIcon} size={16} />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Ghost Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-900">
              Ghost
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="ghost" size="md">
                Save
              </Button>
              <Button variant="ghost" size="md" loading>
                Loading
              </Button>
              <Button variant="ghost" size="md" disabled>
                Disabled
              </Button>
              <Button
                variant="ghost"
                size="md"
                leftIcon={<Icon icon={PlayIcon} size={16} />}
              >
                Play
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Other button types */}
      <div className="space-y-8">
        <h2 className="text-3xl font-semibold text-zinc-900">
          Other button types
        </h2>
        <div className="flex gap-20">
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-800">
              Icon
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="icon" size="icon-lg" aria-label="Add">
                <Icon icon={PlusSignIcon} size={16} />
              </Button>
              <Button
                variant="primary"
                size="icon-lg"
                className="bg-zinc-700"
                aria-label="Add"
              >
                <Icon icon={PlusSignIcon} size={16} />
              </Button>
              <Button variant="icon" size="icon-lg" disabled aria-label="Add">
                <Icon icon={PlusSignIcon} size={16} />
              </Button>
            </div>
          </div>
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-800">
              Link
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="link">Read documentation</Button>
              <Button variant="link" loading>
                Loading link
              </Button>
              <Button variant="link" disabled>
                Disabled link
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
