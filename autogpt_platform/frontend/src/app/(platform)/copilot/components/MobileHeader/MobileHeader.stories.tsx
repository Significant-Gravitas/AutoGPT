import type { Meta, StoryObj } from "@storybook/nextjs";
import { fn } from "@storybook/test";
import { MobileHeader } from "./MobileHeader";

const meta: Meta<typeof MobileHeader> = {
  title: "CoPilot/Chat/MobileHeader",
  component: MobileHeader,
  tags: ["autodocs"],
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Mobile-only header button that opens the session drawer on small screens.",
      },
    },
  },
  args: {
    onOpenDrawer: fn(),
  },
};
export default meta;
type Story = StoryObj<typeof MobileHeader>;

export const Default: Story = {};
