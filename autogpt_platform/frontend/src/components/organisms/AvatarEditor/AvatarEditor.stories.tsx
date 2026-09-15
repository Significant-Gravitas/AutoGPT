import { DEFAULT_CONFIG } from "@/components/molecules/BotAvatar/helpers";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { AvatarEditor } from "./AvatarEditor";

const meta = {
  title: "Organisms/AvatarEditor",
  component: AvatarEditor,
  parameters: { layout: "padded" },
  tags: ["autodocs"],
  args: { value: DEFAULT_CONFIG, onChange: () => {} },
} satisfies Meta<typeof AvatarEditor>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: function DefaultStory(args) {
    const [config, setConfig] = useState(args.value);
    return <AvatarEditor {...args} value={config} onChange={setConfig} />;
  },
};

export const SeededFromAName: Story = {
  args: { name: "Otto" },
  render: function SeededStory(args) {
    const [config, setConfig] = useState(args.value);
    return <AvatarEditor {...args} value={config} onChange={setConfig} />;
  },
};

export const Working: Story = {
  args: { status: "working", size: 220 },
  render: function WorkingStory(args) {
    const [config, setConfig] = useState(args.value);
    return <AvatarEditor {...args} value={config} onChange={setConfig} />;
  },
};
