import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { DryRunToggleButton } from "./DryRunToggleButton";

const meta: Meta<typeof DryRunToggleButton> = {
  title: "Copilot/DryRunToggleButton",
  component: DryRunToggleButton,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Turns test mode on or off for a new chat, with a tooltip explaining the current state.",
      },
    },
  },
  args: {
    onToggle: () => undefined,
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="pt-16">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Off: Story = {
  args: { isDryRun: false },
};

export const On: Story = {
  args: { isDryRun: true },
};
