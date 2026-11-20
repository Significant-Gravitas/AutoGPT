import type { Meta, StoryObj } from "@storybook/nextjs";
import { ExpertIdentityDetails } from "./ExpertIdentityDetails";

const meta = {
  title: "Molecules/ExpertIdentityDetails",
  component: ExpertIdentityDetails,
  args: {
    name: "Jules",
    role: "Social & Content Repurposing",
  },
} satisfies Meta<typeof ExpertIdentityDetails>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Card: Story = {};
export const Page: Story = { args: { size: "page" } };
export const Compact: Story = { args: { size: "compact" } };
export const WithoutArea: Story = {
  args: { role: null, size: "compact" },
};
export const LongArea: Story = {
  args: {
    role: "Social media and content marketing",
    size: "compact",
  },
  decorators: [
    (Story) => (
      <div className="w-48">
        <Story />
      </div>
    ),
  ],
};
