import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  TrackedButton,
  TrackedPrimaryButton,
  TrackedSecondaryButton,
  TrackedDestructiveButton,
  TrackedGhostButton,
} from "./TrackedButton";
import { EventKeys } from "@/services/feature-flags/use-track-event";
import { ArrowRightIcon, PlusIcon } from "@phosphor-icons/react/dist/ssr";

const meta = {
  title: "Components/Atoms/TrackedButton",
  component: TrackedButton,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Button component with built-in LaunchDarkly event tracking. Extends the base Button atom with tracking capabilities.",
      },
    },
  },
  tags: ["autodocs"],
  argTypes: {
    variant: {
      control: { type: "select" },
      options: ["primary", "secondary", "destructive", "outline", "ghost"],
      description: "Visual style variant of the button",
    },
    size: {
      control: { type: "select" },
      options: ["small", "large", "icon"],
      description: "Size of the button",
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
    loading: {
      control: { type: "boolean" },
      description: "Shows loading spinner when true",
    },
    disabled: {
      control: { type: "boolean" },
      description: "Disables the button when true",
    },
  },
} satisfies Meta<typeof TrackedButton>;

export default meta;
type Story = StoryObj<typeof meta>;

// Basic tracked button
export const Default: Story = {
  args: {
    children: "Track Event",
    trackEventKey: "button-clicked",
    trackEventData: { section: "storybook-demo" },
  },
};

// Primary action with tracking
export const PrimaryAction: Story = {
  args: {
    variant: "primary",
    children: "Create Agent",
    trackEventKey: EventKeys.AGENT_CREATED,
    trackEventData: { source: "header" },
    leftIcon: <PlusIcon className="h-4 w-4" />,
  },
};

// Secondary action with tracking
export const SecondaryAction: Story = {
  args: {
    variant: "secondary",
    children: "View Details",
    trackEventKey: "details-viewed",
    trackEventData: { component: "card" },
    rightIcon: <ArrowRightIcon className="h-4 w-4" />,
  },
};

// Danger action with tracking
export const DangerAction: Story = {
  args: {
    variant: "destructive",
    children: "Delete Agent",
    trackEventKey: EventKeys.AGENT_DELETED,
    trackEventData: { confirmationRequired: true },
  },
};

// Ghost button with tracking
export const GhostAction: Story = {
  args: {
    variant: "ghost",
    children: "Cancel",
    trackEventKey: "action-cancelled",
  },
};

// Loading state (tracking still works when loading completes)
export const LoadingState: Story = {
  args: {
    children: "Processing...",
    loading: true,
    trackEventKey: "process-started",
  },
};

// Disabled state (no tracking when disabled)
export const DisabledState: Story = {
  args: {
    children: "Unavailable",
    disabled: true,
    trackEventKey: "disabled-click-attempt",
  },
};

// With numeric metric value
export const WithMetricValue: Story = {
  args: {
    children: "Purchase Credits",
    trackEventKey: EventKeys.CREDITS_PURCHASED,
    trackEventData: { package: "starter" },
    trackMetricValue: 9.99,
  },
};

// Preset components showcase
export const PresetComponents: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <TrackedPrimaryButton
        trackEventKey="primary-action"
        trackEventData={{ preset: true }}
      >
        Primary Preset
      </TrackedPrimaryButton>
      <TrackedSecondaryButton
        trackEventKey="secondary-action"
        trackEventData={{ preset: true }}
      >
        Secondary Preset
      </TrackedSecondaryButton>
      <TrackedDestructiveButton
        trackEventKey="destructive-action"
        trackEventData={{ preset: true }}
      >
        Destructive Preset
      </TrackedDestructiveButton>
      <TrackedGhostButton
        trackEventKey="ghost-action"
        trackEventData={{ preset: true }}
      >
        Ghost Preset
      </TrackedGhostButton>
    </div>
  ),
};

// Complex tracking scenario
export const ComplexTracking: Story = {
  args: {
    children: "Run Agent",
    variant: "primary",
    size: "large",
    trackEventKey: EventKeys.AGENT_RUN_STARTED,
    trackEventData: {
      agentId: "agent-123",
      agentName: "My Agent",
      environment: "production",
      triggeredBy: "manual",
      timestamp: new Date().toISOString(),
    },
    trackMetricValue: 1,
    leftIcon: <ArrowRightIcon className="h-4 w-4" />,
  },
};

// Button as NextLink with tracking
export const AsLink: Story = {
  args: {
    as: "NextLink",
    href: "/marketplace",
    children: "Browse Marketplace",
    trackEventKey: EventKeys.STORE_ACCESSED,
    trackEventData: { referrer: "home" },
  },
};
