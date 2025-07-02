import type { Meta, StoryObj } from "@storybook/nextjs";
import { ExternalLink } from "lucide-react";
import { Link } from "./Link";

const meta: Meta<typeof Link> = {
  title: "Atoms/Link",
  tags: ["autodocs"],
  component: Link,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Link component that wraps Next.js Link with consistent styling. Use `isExternal` prop for external links which automatically adds `target='_blank'` and `rel='noopener noreferrer'` for security.",
      },
    },
  },
  argTypes: {
    href: {
      control: "text",
      description: "The URL or path to link to",
    },
    variant: {
      control: "select",
      options: ["primary", "secondary"],
      description:
        "Link style variant - primary shows underline on hover, secondary always shows underline",
    },
    isExternal: {
      control: "boolean",
      description: "Whether this is an external link (opens in new tab)",
    },
    children: {
      control: "text",
      description: "Link content",
    },
    className: {
      control: "text",
      description: "Additional CSS classes",
    },
  },
  args: {
    href: "/example",
    children: "Add to library",
    variant: "primary",
    isExternal: false,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Primary variant (default - underline on hover)
export const Primary: Story = {
  args: {
    href: "/library",
    children: "Add to library",
    variant: "primary",
    isExternal: false,
  },
};

// Secondary variant (always underlined)
export const Secondary: Story = {
  args: {
    href: "/library",
    children: "Add to library",
    variant: "secondary",
    isExternal: false,
  },
};

// Basic internal link
export const Internal: Story = {
  args: {
    href: "/library",
    children: "Add to library",
    isExternal: false,
  },
};

// External link
export const External: Story = {
  args: {
    href: "https://github.com/Significant-Gravitas/AutoGPT",
    children: "View on GitHub",
    isExternal: true,
  },
};

// Link with icon
export const WithIcon: Story = {
  args: {
    href: "https://docs.autogpt.net",
    children: (
      <span className="inline-flex items-center gap-1">
        Documentation <ExternalLink className="h-3 w-3" />
      </span>
    ),
    isExternal: true,
  },
};

// Variant comparison
export const AllVariants: Story = {
  render: renderVariantComparison,
};

// Render functions as function declarations
function renderVariantComparison() {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4">
        <Link href="/dashboard" variant="primary">
          Primary link
        </Link>
        <Link href="/settings" variant="secondary">
          Secondary link
        </Link>
      </div>
    </div>
  );
}
