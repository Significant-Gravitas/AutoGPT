import { Alert02Icon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ComponentProps } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { AttentionCard } from "./AttentionCard";

type RootProps = ComponentProps<typeof AttentionCard.Root>;

const meta = {
  title: "Organisms/AttentionCard",
  component: AttentionCard.Root,
  parameters: {
    docs: {
      description: {
        component:
          "The composition reference for cards and notices. Build new ones from these slots; pick the look with `variant` and `density` on Root, never class strings at the call site.",
      },
    },
  },
  args: { variant: "default", density: "default" },
  // Icon-only Buttons (the Dismiss slot) label themselves with a Tooltip.
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
  render: (args) => <Example {...args} />,
} satisfies Meta<typeof AttentionCard.Root>;

export default meta;
type Story = StoryObj<typeof meta>;

function Example(props: RootProps) {
  return (
    <div className="max-w-xl">
      <AttentionCard.Root {...props}>
        <AttentionCard.Row align="center">
          <AttentionCard.Icon>
            <Icon icon={Alert02Icon} size={16} />
          </AttentionCard.Icon>
          <AttentionCard.Content>
            <Text variant="body-medium" as="span" className="text-inherit">
              Your weekly report is waiting for approval
            </Text>
          </AttentionCard.Content>
          <AttentionCard.Actions placement="inline">
            <Button variant="secondary" size="small">
              Review
            </Button>
          </AttentionCard.Actions>
          <AttentionCard.Dismiss aria-label="Dismiss" onClick={() => {}} />
        </AttentionCard.Row>
      </AttentionCard.Root>
    </div>
  );
}

// One story per variant.
export const Default: Story = {};
export const Outline: Story = { args: { variant: "outline" } };
export const Elevated: Story = { args: { variant: "elevated" } };
export const Muted: Story = { args: { variant: "muted" } };
export const Featured: Story = { args: { variant: "featured" } };
export const Info: Story = { args: { variant: "info" } };
export const Success: Story = { args: { variant: "success" } };
export const Warning: Story = { args: { variant: "warning" } };
export const WarningStrong: Story = { args: { variant: "warning-strong" } };
export const WarningOutline: Story = { args: { variant: "warning-outline" } };
export const Caution: Story = { args: { variant: "caution" } };
export const ErrorTint: Story = { name: "Error", args: { variant: "error" } };

// One story per density.
export const DensityCompact: Story = { args: { density: "compact" } };
export const DensitySnug: Story = { args: { density: "snug" } };
export const DensityCozy: Story = { args: { density: "cozy" } };
export const DensityDefault: Story = { args: { density: "default" } };
export const DensityComfortable: Story = { args: { density: "comfortable" } };
export const DensityRelaxed: Story = { args: { density: "relaxed" } };
export const DensityRoomy: Story = { args: { density: "roomy" } };
export const DensitySpacious: Story = { args: { density: "spacious" } };
export const DensityFlush: Story = { args: { density: "flush" } };

export const ListWithSeparatorAndCount: Story = {
  args: { variant: "elevated", density: "flush" },
  render: (args) => (
    <div className="max-w-xl">
      <AttentionCard.Root {...args}>
        <AttentionCard.Row align="start" inset>
          <AttentionCard.Content>
            <Text variant="body-medium">Lead Finder</Text>
            <Text variant="body" tone="muted">
              Found 12 new leads overnight
            </Text>
          </AttentionCard.Content>
          <AttentionCard.Count>2 reviews</AttentionCard.Count>
        </AttentionCard.Row>
        <AttentionCard.Separator />
        <AttentionCard.Row align="start" inset>
          <AttentionCard.Content>
            <Text variant="body-medium">Inbox Triage</Text>
            <Text variant="body" tone="muted">
              Sorted 48 emails into 5 folders
            </Text>
          </AttentionCard.Content>
          <AttentionCard.Count>1 review</AttentionCard.Count>
        </AttentionCard.Row>
      </AttentionCard.Root>
    </div>
  ),
};
