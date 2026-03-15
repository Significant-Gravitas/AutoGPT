import type { Meta, StoryObj } from "@storybook/nextjs";
import { ShowMoreText } from "./ShowMoreText";

const meta: Meta<typeof ShowMoreText> = {
  title: "Molecules/ShowMoreText",
  component: ShowMoreText,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component: `
## ShowMoreText Component

A simplified text truncation component that shows a preview of text content with an expand/collapse toggle functionality.

### ✨ Features

- **String content only** - Simplified to only accept string content
- **Text variant integration** - Uses Text component variants for consistent styling
- **Adaptive toggle sizing** - Toggle icons automatically size to match text variant
- **Smart truncation** - Automatically truncates text based on character limit
- **No heading variants** - Only supports body text variants (lead, large, body, small)
- **Inline toggle** - Toggle appears inline at the end of the text
- **TypeScript support** - Full TypeScript interface support

### 🎯 Usage

\`\`\`tsx
<ShowMoreText 
  variant="body" 
  previewLimit={150}
>
  This is a long piece of text that will be truncated at the specified 
  character limit and show a "more" button to expand the full content.
</ShowMoreText>
\`\`\`

### Props

- **children**: String content to show/truncate
- **previewLimit**: Character limit for preview (default: 100)
- **variant**: Text variant to use (excludes heading variants)
- **className**: Additional classes for root container
- **previewClassName**: Additional classes applied in preview mode
- **expandedClassName**: Additional classes applied in expanded mode
- **toggleClassName**: Additional classes for the toggle button
- **defaultExpanded**: Whether to start in expanded state (default: false)

### Supported Text Variants

- **lead** - Large leading text
- **large**, **large-medium**, **large-semibold** - Large body text variants
- **body**, **body-medium** - Standard body text variants
- **small**, **small-medium** - Small text variants
        `,
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    children: {
      control: "text",
      description: "String content to show with truncation",
    },
    previewLimit: {
      control: "number",
      description: "Character limit for preview text",
      table: {
        defaultValue: { summary: "100" },
      },
    },
    variant: {
      control: "select",
      options: [
        "lead",
        "large",
        "large-medium",
        "large-semibold",
        "body",
        "body-medium",
        "small",
        "small-medium",
      ],
      description: "Text variant to use for styling",
      table: {
        defaultValue: { summary: "body" },
      },
    },
    className: {
      control: "text",
      description: "Additional CSS classes for root container",
    },
    toggleClassName: {
      control: "text",
      description: "Additional CSS classes for the toggle button",
    },
    defaultExpanded: {
      control: "boolean",
      description: "Whether to start in expanded state",
      table: {
        defaultValue: { summary: "false" },
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

/**
 * Basic text truncation with default body variant and 100 character limit.
 */
export const Default: Story = {
  args: {
    children:
      "This is a longer piece of text that will be truncated at the preview limit. When you click 'more', you'll see the full content. This demonstrates the basic functionality of the ShowMoreText component with plain text content.",
    variant: "body",
    previewLimit: 100,
  },
};

/**
 * Short text that doesn't need truncation - toggle won't appear.
 */
export const ShortText: Story = {
  args: {
    children: "This text is short enough that no truncation is needed.",
    variant: "body",
    previewLimit: 100,
  },
};

/**
 * Large leading text variant with custom preview limit.
 */
export const LeadVariant: Story = {
  args: {
    children:
      "This example uses the lead text variant which is larger and more prominent. The toggle icons automatically scale to match the text size for a cohesive design.",
    variant: "lead",
    previewLimit: 80,
  },
};

/**
 * Large text variants demonstration.
 */
export const LargeVariants: Story = {
  render: () => (
    <div className="max-w-2xl space-y-4">
      <ShowMoreText variant="large" previewLimit={60}>
        Large variant: This demonstrates how the ShowMoreText component works
        with the large text variant and how the toggle scales appropriately.
      </ShowMoreText>
      <ShowMoreText variant="large-medium" previewLimit={60}>
        Large medium variant: This shows the medium weight version of the large
        text variant with proper toggle sizing.
      </ShowMoreText>
      <ShowMoreText variant="large-semibold" previewLimit={60}>
        Large semibold variant: This demonstrates the semibold version with
        heavier font weight and matching toggle.
      </ShowMoreText>
    </div>
  ),
};

/**
 * Body text variants demonstration.
 */
export const BodyVariants: Story = {
  render: () => (
    <div className="max-w-xl space-y-4">
      <ShowMoreText variant="body" previewLimit={70}>
        Body variant: This is the default text variant used for most content. It
        provides good readability and spacing.
      </ShowMoreText>
      <ShowMoreText variant="body-medium" previewLimit={70}>
        Body medium variant: This uses medium font weight for slightly more
        emphasis while maintaining readability.
      </ShowMoreText>
    </div>
  ),
};

/**
 * Small text variants demonstration.
 */
export const SmallVariants: Story = {
  render: () => (
    <div className="max-w-lg space-y-4">
      <ShowMoreText variant="small" previewLimit={80}>
        Small variant: This demonstrates the small text variant which is useful
        for secondary information, captions, or footnotes where space is
        limited.
      </ShowMoreText>
      <ShowMoreText variant="small-medium" previewLimit={80}>
        Small medium variant: This uses the small size with medium font weight
        for small text that needs slightly more emphasis.
      </ShowMoreText>
    </div>
  ),
};

/**
 * Custom preview limit of 50 characters.
 */
export const CustomLimit: Story = {
  args: {
    children:
      "This example shows how you can customize the preview limit to show more or less text in the initial preview before truncation occurs.",
    variant: "body",
    previewLimit: 50,
  },
};

/**
 * Component that starts in expanded state.
 */
export const DefaultExpanded: Story = {
  args: {
    children:
      "This ShowMoreText component starts in the expanded state by default. You can click 'less' to collapse it to the preview mode. This is useful when you want to show the full content initially but still provide the option to collapse it.",
    variant: "body",
    previewLimit: 80,
    defaultExpanded: true,
  },
};

/**
 * Custom styling for different states.
 */
export const CustomStyling: Story = {
  args: {
    children:
      "This example demonstrates custom styling options. The preview state has a different background color, the expanded state has different padding, and the toggle button has custom styling to match your design system.",
    variant: "body-medium",
    previewLimit: 80,
    className: "max-w-md",
    toggleClassName: "text-blue-600 hover:text-blue-800",
  },
};

/**
 * Very long content to demonstrate with different text sizes.
 */
export const LongContent: Story = {
  render: () => (
    <div className="max-w-2xl space-y-6">
      <div>
        <h3 className="mb-2 text-sm font-medium text-gray-500">Lead Text</h3>
        <ShowMoreText variant="lead" previewLimit={120}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
          eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
          minim veniam, quis nostrud exercitation ullamco laboris nisi ut
          aliquip ex ea commodo consequat. Duis aute irure dolor in
          reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
          pariatur.
        </ShowMoreText>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-gray-500">Body Text</h3>
        <ShowMoreText variant="body" previewLimit={120}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
          eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
          minim veniam, quis nostrud exercitation ullamco laboris nisi ut
          aliquip ex ea commodo consequat. Duis aute irure dolor in
          reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
          pariatur.
        </ShowMoreText>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-gray-500">Small Text</h3>
        <ShowMoreText variant="small" previewLimit={120}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
          eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
          minim veniam, quis nostrud exercitation ullamco laboris nisi ut
          aliquip ex ea commodo consequat. Duis aute irure dolor in
          reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
          pariatur.
        </ShowMoreText>
      </div>
    </div>
  ),
};
