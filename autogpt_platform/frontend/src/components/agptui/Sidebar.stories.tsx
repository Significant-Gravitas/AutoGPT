import type { Meta, StoryObj } from "@storybook/nextjs";
import { Sidebar } from "./Sidebar";

const meta = {
  title: "Legacy/Sidebar",
  component: Sidebar,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    linkGroups: { control: "object" },
  },
} satisfies Meta<typeof Sidebar>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultLinkGroups = [
  {
    links: [
      { text: "Agent dashboard", href: "/dashboard" },
      { text: "Integrations", href: "/integrations" },
      { text: "Profile", href: "/profile" },
      { text: "Settings", href: "/settings" },
    ],
  },
];

export const Default: Story = {
  args: {
    linkGroups: defaultLinkGroups,
  },
};
