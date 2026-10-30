import type { Meta, StoryObj } from "@storybook/nextjs";
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
import { TooltipProvider } from "../Tooltip/BaseTooltip";
import { Icon } from "../Icon/Icon";
import { Text } from "../Text/Text";
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
    layout: "centered",
    docs: {
      description: {
        component:
          "Button component with multiple variants and sizes based on our design system. Built on top of shadcn/ui button with custom styling.",
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
      options: ["small", "large", "icon", "xs", "icon-xs", "icon-sm"],
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
    size: "large",
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
          '`size="xs"` is the 28px rounded-rectangle chip used for row and header actions across Home and Team. Pair with `leadingIcon` for a 14px Hugeicon.',
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
          '`size="icon-xs"` (28px) and `size="icon-sm"` (32px) are square icon-only actions. `aria-label` is required and doubles as the tooltip. `variant="floating"` sits over images and card covers.',
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
    leftIcon: <Icon icon={PlayIcon} className="h-4 w-4" />,
    children: "Play",
  },
};

export const WithRightIcon: Story = {
  args: {
    variant: "outline",
    rightIcon: <Icon icon={PlusSignIcon} className="h-4 w-4" />,
    children: "Add Item",
  },
};

export const IconOnly: Story = {
  args: {
    variant: "icon",
    size: "icon",
    children: <Icon icon={PlusSignIcon} className="h-4 w-4" />,
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

// Every variant at every size, at rest, disabled and loading.
export const VariantMatrix: Story = {
  render: renderVariantMatrix,
};

// The rounded-md family for dialogs, forms and toggle rows that keep the
// pre-pill look; what the legacy-* variants became.
export const CompactVariants: Story = {
  render: renderCompactVariants,
};

// Render functions as function declarations
const COMPACT_VARIANTS = [
  "compact-primary",
  "compact-destructive",
  "compact-outline",
  "compact-ghost",
] as const;

const COMPACT_SIZES = [
  "compact",
  "compact-sm",
  "compact-icon",
  "compact-icon-sm",
] as const;

function renderCompactVariants() {
  return (
    <div className="space-y-6 p-8">
      {COMPACT_VARIANTS.map((variant) => (
        <div key={variant} className="space-y-2">
          <Text variant="small-medium" tone="muted">
            {variant}
          </Text>
          <div className="flex flex-wrap items-center gap-3">
            {COMPACT_SIZES.map((size) => (
              <Button key={size} variant={variant} size={size}>
                {size.startsWith("compact-icon") ? (
                  <Icon icon={PlusSignIcon} size={16} />
                ) : (
                  size
                )}
              </Button>
            ))}
            <Button variant={variant} size="compact" disabled>
              Disabled
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}

const MATRIX_VARIANTS = [
  "primary",
  "secondary",
  "destructive",
  "outline",
  "ghost",
  "icon",
  "toggle",
  "floating",
  "link",
] as const;

const MATRIX_SIZES = [
  "small",
  "large",
  "icon",
  "xs",
  "icon-xs",
  "icon-sm",
] as const;

function renderVariantMatrix() {
  return (
    <div className="space-y-6 p-8">
      {MATRIX_VARIANTS.map((variant) => (
        <div key={variant} className="space-y-2">
          <Text variant="small-medium" tone="muted">
            {variant}
          </Text>
          <div className="flex flex-wrap items-center gap-3">
            {MATRIX_SIZES.map((size) => (
              <Button
                key={size}
                variant={variant}
                size={size}
                leadingIcon={PlusSignIcon}
                aria-label={`${variant} ${size}`}
              >
                {size.startsWith("icon") ? null : size}
              </Button>
            ))}
            <Button variant={variant} disabled>
              Disabled
            </Button>
            <Button variant={variant} loading>
              Loading
            </Button>
            <Button variant={variant} aria-pressed>
              Pressed
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}

function renderActionButtons() {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="xs">
          Chat
        </Button>
        <Button variant="secondary" size="xs">
          Manage
        </Button>
        <Button variant="outline" size="xs">
          New Pod
        </Button>
        <Button variant="ghost" size="xs">
          Open in library
        </Button>
        <Button variant="destructive" size="xs">
          Fire Maria
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="xs" leadingIcon={SparklesIcon}>
          New Expert
        </Button>
        <Button variant="secondary" size="xs" leadingIcon={PencilEdit02Icon}>
          Edit Soul
        </Button>
        <Button variant="secondary" size="xs" leadingIcon={PlusSignIcon}>
          Install workflow
        </Button>
        <Button
          variant="ghost"
          size="xs"
          leadingIcon={FilterHorizontalIcon}
          className="text-foreground-secondary"
        >
          All work
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="secondary" size="xs" loading>
          Resuming...
        </Button>
        <Button variant="secondary" size="xs" disabled>
          Disabled
        </Button>
        <Button as="NextLink" href="#" variant="secondary" size="xs">
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
          size="icon-xs"
          leadingIcon={FilterHorizontalIcon}
          aria-label="Filter workflows"
        />
        <Button
          variant="primary"
          size="icon-xs"
          leadingIcon={Tick02Icon}
          aria-label="Approve"
        />
        <Button
          variant="destructive"
          size="icon-xs"
          leadingIcon={Cancel01Icon}
          aria-label="Confirm decline"
        />
        <Button
          variant="ghost"
          size="icon-xs"
          leadingIcon={MoreHorizontalIcon}
          aria-label="More actions"
        />
      </div>
      <div className="flex flex-wrap items-center gap-2 rounded-lg bg-gradient-to-r from-violet-200 to-sky-200 p-4">
        <Button
          variant="floating"
          size="icon-xs"
          leadingIcon={PencilEdit02Icon}
          aria-label="Edit workflow"
        />
        <Button
          variant="floating"
          size="icon-xs"
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
      <Button variant="toggle" size="xs" aria-pressed={false}>
        Needs review (3)
      </Button>
      <Button
        variant="toggle"
        size="xs"
        aria-pressed
        className="border-border-base bg-surface aria-pressed:border-warning-border aria-pressed:bg-warning-surface aria-pressed:text-warning-foreground"
      >
        Needs review (3)
      </Button>
      <div className="flex h-7 items-center rounded-md border border-border-base p-0.5">
        <Button
          variant="toggle"
          size="icon-xs"
          className="size-6 rounded"
          leadingIcon={ListViewIcon}
          aria-label="List view"
          aria-pressed
        />
        <Button
          variant="toggle"
          size="icon-xs"
          className="size-6 rounded"
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
        <h3 className="mb-4 text-base font-medium text-foreground-strong">
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
        <h3 className="mb-4 text-base font-medium text-error-foreground">
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
        <p className="mt-2 text-sm text-foreground-secondary">
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
      <Button variant="primary" size="small">
        Primary
      </Button>
      <Button variant="secondary" size="small">
        Secondary
      </Button>
      <Button variant="destructive" size="small">
        Delete
      </Button>
      <Button variant="outline" size="small">
        Outline
      </Button>
      <Button variant="ghost" size="small">
        Ghost
      </Button>
    </div>
  );
}

function renderLargeButtons() {
  return (
    <div className="flex flex-wrap gap-4">
      <Button variant="primary" size="large">
        Primary
      </Button>
      <Button variant="secondary" size="large">
        Secondary
      </Button>
      <Button variant="destructive" size="large">
        Delete
      </Button>
      <Button variant="outline" size="large">
        Outline
      </Button>
      <Button variant="ghost" size="large">
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
        <h2 className="text-3xl font-semibold text-foreground-strong">
          Large buttons
        </h2>
        <div className="flex flex-wrap gap-20">
          {/* Primary */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Primary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="primary" size="large">
                Save
              </Button>
              <Button variant="primary" size="large" loading>
                Loading
              </Button>
              <Button variant="primary" size="large" disabled>
                Disabled
              </Button>
              <Button
                variant="primary"
                size="large"
                leftIcon={<Icon icon={PlayIcon} className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Secondary */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Secondary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="secondary" size="large">
                Save
              </Button>
              <Button variant="secondary" size="large" loading>
                Loading
              </Button>
              <Button variant="secondary" size="large" disabled>
                Disabled
              </Button>
              <Button
                variant="secondary"
                size="large"
                leftIcon={<Icon icon={PlayIcon} className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Destructive */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Destructive
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="destructive" size="large">
                Save
              </Button>
              <Button variant="destructive" size="large" loading>
                Loading
              </Button>
              <Button variant="destructive" size="large" disabled>
                Disabled
              </Button>
              <Button
                variant="destructive"
                size="large"
                leftIcon={<Icon icon={PlayIcon} className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Outline */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Outline
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="outline" size="large">
                Save
              </Button>
              <Button variant="outline" size="large" loading>
                Loading
              </Button>
              <Button variant="outline" size="large" disabled>
                Disabled
              </Button>
              <Button
                variant="outline"
                size="large"
                leftIcon={<Icon icon={PlayIcon} className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Ghost */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Save
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="ghost" size="large">
                Text
              </Button>
              <Button variant="ghost" size="large" loading>
                Loading
              </Button>
              <Button variant="ghost" size="large" disabled>
                Disabled
              </Button>
              <Button
                variant="ghost"
                size="large"
                leftIcon={<Icon icon={PlayIcon} className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Small buttons section */}
      <div className="space-y-8">
        <h2 className="text-3xl font-semibold text-foreground-strong">
          Small buttons
        </h2>
        <div className="flex flex-wrap gap-20">
          {/* Primary Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Primary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="primary" size="small">
                Save
              </Button>
              <Button variant="primary" size="small" loading>
                Loading
              </Button>
              <Button variant="primary" size="small" disabled>
                Disabled
              </Button>
              <Button
                variant="primary"
                size="small"
                leftIcon={<Icon icon={PlayIcon} className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Secondary Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Secondary
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="secondary" size="small">
                Save
              </Button>
              <Button variant="secondary" size="small" loading>
                Loading
              </Button>
              <Button variant="secondary" size="small" disabled>
                Disabled
              </Button>
              <Button
                variant="secondary"
                size="small"
                leftIcon={<Icon icon={PlayIcon} className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Destructive Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Destructive
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="destructive" size="small">
                Save
              </Button>
              <Button variant="destructive" size="small" loading>
                Loading
              </Button>
              <Button variant="destructive" size="small" disabled>
                Disabled
              </Button>
              <Button
                variant="destructive"
                size="small"
                leftIcon={<Icon icon={PlayIcon} className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Outline Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Outline
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="outline" size="small">
                Save
              </Button>
              <Button variant="outline" size="small" loading>
                Loading
              </Button>
              <Button variant="outline" size="small" disabled>
                Disabled
              </Button>
              <Button
                variant="outline"
                size="small"
                leftIcon={<Icon icon={PlayIcon} className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Ghost Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-strong">
              Ghost
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="ghost" size="small">
                Save
              </Button>
              <Button variant="ghost" size="small" loading>
                Loading
              </Button>
              <Button variant="ghost" size="small" disabled>
                Disabled
              </Button>
              <Button
                variant="ghost"
                size="small"
                leftIcon={<Icon icon={PlayIcon} className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Other button types */}
      <div className="space-y-8">
        <h2 className="text-3xl font-semibold text-foreground-strong">
          Other button types
        </h2>
        <div className="flex gap-20">
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-emphasis">
              Icon
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="icon" size="icon">
                <Icon icon={PlusSignIcon} className="h-4 w-4" />
              </Button>
              <Button variant="primary" size="icon" className="bg-action">
                <Icon icon={PlusSignIcon} className="h-4 w-4" />
              </Button>
              <Button variant="icon" size="icon" disabled>
                <Icon icon={PlusSignIcon} className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-foreground-emphasis">
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
