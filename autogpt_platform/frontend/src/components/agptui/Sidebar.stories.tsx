import type { Meta, StoryObj } from "@storybook/react";
import { Sidebar } from "./Sidebar";

const meta = {
  title: "AGPT UI/Sidebar",
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
    links: [{ text: "Integrations", href: "/integrations" }],
  },
  {
    links: [
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

export const SingleGroup: Story = {
  args: {
    linkGroups: [defaultLinkGroups[0]],
  },
};

export const ManyGroups: Story = {
  args: {
    linkGroups: [
      ...defaultLinkGroups,
      {
        links: [
          { text: "About", href: "/about" },
          { text: "Contact", href: "/contact" },
        ],
      },
      {
        links: [
          { text: "Terms", href: "/terms" },
          { text: "Privacy", href: "/privacy" },
        ],
      },
    ],
  },
};

export const LongLinkTexts: Story = {
  args: {
    linkGroups: [
      {
        links: [
          {
            text: "This is a very long link text that might wrap",
            href: "/long-link-1",
          },
          {
            text: "Another extremely long link text for testing purposes",
            href: "/long-link-2",
          },
        ],
      },
      ...defaultLinkGroups,
    ],
  },
};
