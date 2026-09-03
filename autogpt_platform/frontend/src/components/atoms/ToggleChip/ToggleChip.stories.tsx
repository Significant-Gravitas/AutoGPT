import { Icon } from "@/components/atoms/Icon/Icon";
import { TooltipProvider } from "@/components/ui/tooltip";
import { BrainIcon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { ToggleChip } from "./ToggleChip";

const meta: Meta<typeof ToggleChip> = {
  title: "Atoms/ToggleChip",
  component: ToggleChip,
  tags: ["autodocs"],
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
          "A pill-shaped toggle with an icon, label, and tooltip. Hovering swaps the icon for a toggle glyph, and the label blurs between states. Locked chips keep their icon and refuse the hover affordance.",
      },
    },
  },
  argTypes: {
    label: { control: "text", description: "Visible chip label." },
    tooltip: { control: "text", description: "Tooltip content." },
    ariaLabel: { control: "text", description: "Accessible button label." },
    pressed: { control: "boolean", description: "Pressed state (controlled)." },
    locked: { control: "boolean", description: "Disable toggling." },
    className: { control: "text", description: "Optional className." },
    onToggle: { action: "toggle", description: "Toggle handler." },
  },
  args: {
    icon: <Icon icon={BrainIcon} size={14} />,
    label: "Extended Thinking",
    tooltip: "Switch copilot mode",
    ariaLabel: "Toggle Extended Thinking",
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Basic: Story = {
  render: function BasicStory(args) {
    const [pressed, setPressed] = useState<boolean>(false);
    return (
      <ToggleChip
        {...args}
        pressed={pressed}
        onToggle={() => {
          setPressed(!pressed);
          if (args.onToggle) args.onToggle();
        }}
      />
    );
  },
};

export const Locked: Story = {
  args: {
    pressed: true,
    locked: true,
    tooltip: "Locked to Extended Thinking",
    onToggle: () => {},
  },
};
