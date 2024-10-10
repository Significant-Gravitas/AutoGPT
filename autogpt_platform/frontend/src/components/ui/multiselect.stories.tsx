import React from "react";
import type { Meta, StoryObj } from "@storybook/react";

import {
  MultiSelector,
  MultiSelectorTrigger,
  MultiSelectorInput,
  MultiSelectorContent,
  MultiSelectorList,
  MultiSelectorItem,
} from "./multiselect";

const meta = {
  title: "UI/MultiSelector",
  component: MultiSelector,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    loop: {
      control: "boolean",
    },
    values: {
      control: "object",
    },
    onValuesChange: { action: "onValuesChange" },
  },
} satisfies Meta<typeof MultiSelector>;

export default meta;
type Story = StoryObj<typeof meta>;

const MultiSelectorExample = (args: any) => {
  const [values, setValues] = React.useState<string[]>(args.values || []);

  return (
    <MultiSelector values={values} onValuesChange={setValues} {...args}>
      <MultiSelectorTrigger>
        <MultiSelectorInput placeholder="Select items..." />
      </MultiSelectorTrigger>
      <MultiSelectorContent>
        <MultiSelectorList>
          <MultiSelectorItem value="apple">Apple</MultiSelectorItem>
          <MultiSelectorItem value="banana">Banana</MultiSelectorItem>
          <MultiSelectorItem value="cherry">Cherry</MultiSelectorItem>
          <MultiSelectorItem value="date">Date</MultiSelectorItem>
          <MultiSelectorItem value="elderberry">Elderberry</MultiSelectorItem>
        </MultiSelectorList>
      </MultiSelectorContent>
    </MultiSelector>
  );
};

export const Default: Story = {
  render: (args) => <MultiSelectorExample {...args} />,
  args: {
    values: [],
    onValuesChange: (value: string[]) => {},
  },
};

export const WithLoop: Story = {
  render: (args) => <MultiSelectorExample {...args} />,
  args: {
    values: [],
    onValuesChange: (value: string[]) => {},
    loop: true,
  },
};

export const WithInitialValues: Story = {
  render: (args) => <MultiSelectorExample {...args} />,
  args: {
    values: ["apple", "banana"],
    onValuesChange: (value: string[]) => {},
  },
};

export const WithDisabledItem: Story = {
  render: (args) => (
    <MultiSelectorExample {...args}>
      <MultiSelectorTrigger>
        <MultiSelectorInput placeholder="Select items..." />
      </MultiSelectorTrigger>
      <MultiSelectorContent>
        <MultiSelectorList>
          <MultiSelectorItem value="apple">Apple</MultiSelectorItem>
          <MultiSelectorItem value="banana">Banana</MultiSelectorItem>
          <MultiSelectorItem value="cherry" disabled>
            Cherry (Disabled)
          </MultiSelectorItem>
          <MultiSelectorItem value="date">Date</MultiSelectorItem>
          <MultiSelectorItem value="elderberry">Elderberry</MultiSelectorItem>
        </MultiSelectorList>
      </MultiSelectorContent>
    </MultiSelectorExample>
  ),
  args: {
    values: [],
    onValuesChange: (value: string[]) => {},
  },
};
