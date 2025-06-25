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
    isExternal: false,
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

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

// Different link types
export const LinkTypes: Story = {
  render: renderLinkTypes,
};

// Links in context
export const InContext: Story = {
  render: renderInContext,
};

// Render functions as function declarations
function renderLinkTypes() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold">Internal Links</h3>
        <div className="flex flex-wrap gap-4">
          <Link href="/dashboard">Dashboard</Link>
          <Link href="/settings">Settings</Link>
          <Link href="/profile">Profile</Link>
          <Link href="/library">Add to library</Link>
        </div>
      </div>

      <div className="space-y-2">
        <h3 className="text-lg font-semibold">External Links</h3>
        <div className="flex flex-wrap gap-4">
          <Link
            href="https://github.com/Significant-Gravitas/AutoGPT"
            isExternal
          >
            GitHub Repository
          </Link>
          <Link href="https://docs.autogpt.net" isExternal>
            Documentation
          </Link>
          <Link href="https://discord.gg/autogpt" isExternal>
            Discord Community
          </Link>
        </div>
      </div>

      <div className="space-y-2">
        <h3 className="text-lg font-semibold">Links with Icons</h3>
        <div className="flex flex-wrap gap-4">
          <Link href="https://docs.autogpt.net" isExternal>
            <span className="inline-flex items-center gap-1">
              Documentation <ExternalLink className="h-3 w-3" />
            </span>
          </Link>
          <Link href="https://github.com" isExternal>
            <span className="inline-flex items-center gap-1">
              GitHub <ExternalLink className="h-3 w-3" />
            </span>
          </Link>
        </div>
      </div>
    </div>
  );
}

function renderInContext() {
  return (
    <div className="max-w-2xl space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">In Paragraph Text</h3>
        <p className="text-sm text-gray-600">
          This is a paragraph with an{" "}
          <Link href="/internal-page">internal link</Link> and an{" "}
          <Link href="https://example.com" isExternal>
            external link
          </Link>{" "}
          to demonstrate how links appear in flowing text. The styling is
          consistent with our design system.
        </p>
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-semibold">In Navigation</h3>
        <nav className="flex space-x-6">
          <Link href="/dashboard">Dashboard</Link>
          <Link href="/agents">Agents</Link>
          <Link href="/library">Library</Link>
          <Link href="/settings">Settings</Link>
        </nav>
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-semibold">In Cards</h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-2 rounded-lg border p-4">
            <h4 className="font-medium">Agent Template</h4>
            <p className="text-sm text-gray-600">
              A powerful automation template for your workflow.
            </p>
            <Link href="/templates/agent">Add to library</Link>
          </div>
          <div className="space-y-2 rounded-lg border p-4">
            <h4 className="font-medium">External Resource</h4>
            <p className="text-sm text-gray-600">
              Learn more about this topic from our documentation.
            </p>
            <Link href="https://docs.autogpt.net/guides" isExternal>
              <span className="inline-flex items-center gap-1">
                Read Guide <ExternalLink className="h-3 w-3" />
              </span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
