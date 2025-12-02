import type { Meta, StoryObj } from "@storybook/nextjs";
import { Play, Plus } from "lucide-react";
import { Button } from "./Button";

const meta: Meta<typeof Button> = {
  title: "Atoms/Button",
  tags: ["autodocs"],
  component: Button,
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
        "link",
        "loading",
      ],
      description: "Button style variant",
    },
    size: {
      control: "select",
      options: ["small", "large", "icon"],
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

// With icons
export const WithLeftIcon: Story = {
  args: {
    variant: "primary",
    leftIcon: <Play className="h-4 w-4" />,
    children: "Play",
  },
};

export const WithRightIcon: Story = {
  args: {
    variant: "outline",
    rightIcon: <Plus className="h-4 w-4" />,
    children: "Add Item",
  },
};

export const IconOnly: Story = {
  args: {
    variant: "icon",
    size: "icon",
    children: <Plus className="h-4 w-4" />,
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
function renderContextualLoadingExamples() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-4 text-base font-medium text-neutral-900">
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
        <p className="mt-2 text-sm text-neutral-600">
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
        <h2 className="text-3xl font-semibold text-neutral-900">
          Large buttons
        </h2>
        <div className="flex flex-wrap gap-20">
          {/* Primary */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Secondary */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Destructive */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Outline */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Ghost */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-5 w-5" />}
              >
                Play
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Small buttons section */}
      <div className="space-y-8">
        <h2 className="text-3xl font-semibold text-neutral-900">
          Small buttons
        </h2>
        <div className="flex flex-wrap gap-20">
          {/* Primary Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Secondary Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Destructive Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Outline Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>

          {/* Ghost Small */}
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-neutral-900">
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
                leftIcon={<Play className="h-4 w-4" />}
              >
                Play
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Other button types */}
      <div className="space-y-8">
        <h2 className="text-3xl font-semibold text-neutral-900">
          Other button types
        </h2>
        <div className="flex gap-20">
          <div className="flex flex-col gap-5">
            <div className="font-['Geist'] text-base font-medium text-zinc-800">
              Icon
            </div>
            <div className="flex flex-col gap-8">
              <Button variant="icon" size="icon">
                <Plus className="h-4 w-4" />
              </Button>
              <Button variant="primary" size="icon" className="bg-zinc-700">
                <Plus className="h-4 w-4" />
              </Button>
              <Button variant="icon" size="icon" disabled>
                <Plus className="h-4 w-4" />
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
