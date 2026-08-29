import type { Meta, StoryObj } from "@storybook/nextjs";
import { useEffect, useState } from "react";

import { SearchInput } from "./SearchInput";

const meta: Meta<typeof SearchInput> = {
  title: "Molecules/SearchInput",
  component: SearchInput,
  parameters: {
    layout: "centered",
  },
  args: {
    placeholder: "Search",
    size: "medium",
  },
  argTypes: {
    size: {
      control: { type: "radio" },
      options: ["small", "medium"],
    },
    disabled: { control: "boolean" },
  },
};

export default meta;
type Story = StoryObj<typeof SearchInput>;

function ControlledExample(args: React.ComponentProps<typeof SearchInput>) {
  const [value, setValue] = useState(args.value ?? "");

  useEffect(() => {
    setValue(args.value ?? "");
  }, [args.value]);

  return (
    <div className="w-[360px]">
      <SearchInput {...args} value={value} onChange={setValue} />
    </div>
  );
}

export const Default: Story = {
  render: (args) => <ControlledExample {...args} />,
};

export const WithValue: Story = {
  args: { value: "invoice agent" },
  render: (args) => <ControlledExample {...args} />,
};

export const Small: Story = {
  args: { size: "small", value: "scraper" },
  render: (args) => <ControlledExample {...args} />,
};

export const Disabled: Story = {
  args: { value: "agent", disabled: true },
  render: (args) => <ControlledExample {...args} />,
};
