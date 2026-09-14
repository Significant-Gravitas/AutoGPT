import type { Meta, StoryObj } from "@storybook/nextjs";
import { ProviderBox } from "./ProviderBox";

const meta = {
  title: "Integrations/Subscription Provider",
  component: ProviderBox,
  parameters: { layout: "centered" },
  args: {
    name: "Microsoft 365 Copilot",
    logoSrc: "/integrations/microsoft.webp",
    state: "available",
  },
  decorators: [
    (Story) => (
      <div className="w-48">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ProviderBox>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Available: Story = {};
export const Connected: Story = { args: { state: "connected" } };
export const Connecting: Story = { args: { isBusy: true } };
export const ComingSoon: Story = {
  args: {
    name: "GitHub Copilot",
    logoSrc: "/integrations/github.png",
    state: "coming-soon",
  },
};
