import { PlusIcon } from "@phosphor-icons/react/dist/ssr";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { SplitButton } from "./SplitButton";

const meta: Meta<typeof SplitButton> = {
  title: "Atoms/SplitButton",
  component: SplitButton,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "A primary action button joined to a caret that opens a menu of alternate actions. Composes the Button atom and DropdownMenu molecule so it reads as one control.",
      },
    },
  },
  argTypes: {
    variant: {
      control: "select",
      options: ["primary", "secondary", "ghost", "outline", "destructive"],
      description: "Button variant applied to both segments.",
    },
    size: {
      control: "select",
      options: ["small", "large"],
      description: "Button size applied to both segments.",
    },
    loading: {
      control: "boolean",
      description: "Show a spinner on the primary segment.",
    },
    disabled: { control: "boolean", description: "Disable both segments." },
    primaryLabel: {
      control: "text",
      description: "Label for the primary action.",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

const sampleItems = [
  { key: "org", label: "Organization", onSelect: () => {} },
  { key: "team-a", label: "Design team", onSelect: () => {} },
  { key: "team-b", label: "Growth team", onSelect: () => {} },
];

export const Basic: Story = {
  args: {
    primaryLabel: "Add to Organization",
    variant: "primary",
    size: "large",
    items: sampleItems,
  },
  render: function BasicStory(args) {
    return <SplitButton {...args} onPrimaryClick={() => {}} />;
  },
};

export const Ghost: Story = {
  args: {
    primaryLabel: "Add to Design team",
    variant: "ghost",
    size: "small",
    leftIcon: <PlusIcon size={14} weight="bold" />,
    items: sampleItems,
  },
  render: function GhostStory(args) {
    return <SplitButton {...args} onPrimaryClick={() => {}} />;
  },
};

export const Loading: Story = {
  args: {
    primaryLabel: "Adding...",
    variant: "primary",
    size: "large",
    loading: true,
    items: sampleItems,
  },
  render: function LoadingStory(args) {
    return <SplitButton {...args} onPrimaryClick={() => {}} />;
  },
};

export const Disabled: Story = {
  args: {
    primaryLabel: "Add to Organization",
    variant: "secondary",
    size: "large",
    disabled: true,
    items: sampleItems,
  },
  render: function DisabledStory(args) {
    return <SplitButton {...args} onPrimaryClick={() => {}} />;
  },
};
