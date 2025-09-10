import type { Meta, StoryObj } from "@storybook/nextjs";
import { TrackedLink } from "./TrackedLink";
import { EventKeys } from "@/services/feature-flags/use-track-event";

const meta = {
  title: "Components/Atoms/TrackedLink",
  component: TrackedLink,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Link component with built-in LaunchDarkly event tracking. Extends the base Link atom with tracking capabilities.",
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    href: {
      control: { type: "text" },
      description: "URL or path for the link",
    },
    variant: {
      control: { type: "select" },
      options: ["primary", "secondary"],
      description: "Visual style variant of the link",
    },
    isExternal: {
      control: { type: "boolean" },
      description: "Opens link in new tab when true",
    },
    trackEventKey: {
      control: { type: "text" },
      description: "LaunchDarkly event key to track on click",
    },
    trackEventData: {
      control: { type: "object" },
      description: "Additional data to send with the tracking event",
    },
    trackMetricValue: {
      control: { type: "number" },
      description: "Numeric value for metrics tracking",
    },
  },
} satisfies Meta<typeof TrackedLink>;

export default meta;
type Story = StoryObj<typeof meta>;

// Basic tracked link
export const Default: Story = {
  args: {
    href: "/dashboard",
    children: "Go to Dashboard",
    trackEventKey: "dashboard-link-clicked",
    trackEventData: { source: "storybook" },
  },
};

// Primary variant with tracking
export const Primary: Story = {
  args: {
    href: "/marketplace",
    variant: "primary",
    children: "Browse Marketplace",
    trackEventKey: EventKeys.STORE_ACCESSED,
    trackEventData: { referrer: "navigation" },
  },
};

// Secondary variant with tracking
export const Secondary: Story = {
  args: {
    href: "/help",
    variant: "secondary",
    children: "Need help?",
    trackEventKey: "help-link-clicked",
    trackEventData: { context: "footer" },
  },
};

// External link with tracking
export const ExternalLink: Story = {
  args: {
    href: "https://docs.autogpt.com",
    isExternal: true,
    children: "View Documentation",
    trackEventKey: "documentation-link-clicked",
    trackEventData: {
      destination: "external-docs",
      openInNewTab: true,
    },
  },
};

// Link with numeric metric
export const WithMetricValue: Story = {
  args: {
    href: "/premium",
    children: "Upgrade to Premium",
    trackEventKey: "premium-link-clicked",
    trackEventData: { plan: "premium" },
    trackMetricValue: 29.99,
  },
};

// Complex tracking scenario
export const ComplexTracking: Story = {
  args: {
    href: "/agent/123",
    children: "View Agent Details",
    trackEventKey: EventKeys.STORE_AGENT_VIEWED,
    trackEventData: {
      agentId: "123",
      agentName: "Sample Agent",
      source: "search-results",
      position: 3,
      timestamp: new Date().toISOString(),
    },
  },
};

// Multiple links showcase
export const MultipleLinks: Story = {
  args: {
    href: "/",
    children: "Link",
  },
  render: () => (
    <div className="flex flex-col gap-4">
      <TrackedLink
        href="/agents"
        trackEventKey="agents-nav-clicked"
        trackEventData={{ section: "header" }}
      >
        My Agents
      </TrackedLink>
      <TrackedLink
        href="/marketplace"
        variant="secondary"
        trackEventKey={EventKeys.STORE_ACCESSED}
        trackEventData={{ section: "header" }}
      >
        Marketplace
      </TrackedLink>
      <TrackedLink
        href="https://github.com/Significant-Gravitas/AutoGPT"
        isExternal
        trackEventKey="github-link-clicked"
        trackEventData={{ repository: "AutoGPT" }}
      >
        View on GitHub
      </TrackedLink>
      <TrackedLink
        href="/support"
        variant="primary"
        trackEventKey="support-link-clicked"
        trackEventData={{ urgency: "normal" }}
      >
        Contact Support
      </TrackedLink>
    </div>
  ),
};

// Navigation menu example
export const NavigationMenu: Story = {
  args: {
    href: "/",
    children: "Link",
  },
  render: () => (
    <nav className="flex gap-6">
      <TrackedLink
        href="/"
        trackEventKey="nav-home-clicked"
        trackEventData={{ menu: "main" }}
      >
        Home
      </TrackedLink>
      <TrackedLink
        href="/agents"
        trackEventKey="nav-agents-clicked"
        trackEventData={{ menu: "main" }}
      >
        Agents
      </TrackedLink>
      <TrackedLink
        href="/marketplace"
        trackEventKey={EventKeys.STORE_ACCESSED}
        trackEventData={{ menu: "main", source: "navigation" }}
      >
        Marketplace
      </TrackedLink>
      <TrackedLink
        href="/settings"
        trackEventKey="nav-settings-clicked"
        trackEventData={{ menu: "main" }}
      >
        Settings
      </TrackedLink>
    </nav>
  ),
};
