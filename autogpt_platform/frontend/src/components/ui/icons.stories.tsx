import type { Meta, StoryObj } from "@storybook/react";
import {
  IconUser,
  IconUserPlus,
  IconKey,
  IconKeyPlus,
  IconWorkFlow,
  IconPlay,
  IconSquare,
  IconSquareActivity,
  IconRefresh,
  IconSave,
  IconUndo2,
  IconRedo2,
  IconToyBrick,
  IconCircleAlert,
  IconCircleUser,
  IconPackage2,
  IconMegaphone,
  IconMenu,
  IconCoin,
} from "./icons";

const meta = {
  title: "UI/Icons",
  component: IconUser, // Add a component property
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    size: {
      control: "select",
      options: ["default", "sm", "lg"],
    },
    className: { control: "text" },
  },
} satisfies Meta<typeof IconUser>; // Specify the type parameter

export default meta;
type Story = StoryObj<typeof meta>;

const IconWrapper = ({ children }: { children: React.ReactNode }) => (
  <div className="flex flex-wrap gap-4">{children}</div>
);

export const AllIcons: Story = {
  render: (args) => (
    <IconWrapper>
      <IconUser {...args} />
      <IconUserPlus {...args} />
      <IconKey {...args} />
      <IconKeyPlus {...args} />
      <IconWorkFlow {...args} />
      <IconPlay {...args} />
      <IconSquare {...args} />
      <IconSquareActivity {...args} />
      <IconRefresh {...args} />
      <IconSave {...args} />
      <IconUndo2 {...args} />
      <IconRedo2 {...args} />
      <IconToyBrick {...args} />
      <IconCircleAlert {...args} />
      <IconCircleUser {...args} />
      <IconPackage2 {...args} />
      <IconMegaphone {...args} />
      <IconMenu {...args} />
      <IconCoin {...args} />
    </IconWrapper>
  ),
};

export const DefaultSize: Story = {
  args: {
    size: "default",
  },
};

export const SmallSize: Story = {
  args: {
    size: "sm",
  },
};

export const LargeSize: Story = {
  args: {
    size: "lg",
  },
};

export const CustomColor: Story = {
  args: {
    className: "text-blue-500",
  },
};
