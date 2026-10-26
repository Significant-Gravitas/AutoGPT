import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorItem,
  MultiSelectorList,
  MultiSelectorTrigger,
} from "./MultiSelect";

const meta: Meta<typeof MultiSelector> = {
  title: "Molecules/MultiSelect",
  component: MultiSelector,
  parameters: { layout: "centered" },
};

export default meta;
type Story = StoryObj<typeof MultiSelector>;

const FRUIT = ["Apples", "Bananas", "Cherries", "Dates", "Elderberries"];

interface Props {
  initial: string[];
  disabled?: string;
}

function Picker({ initial, disabled }: Props) {
  const [values, setValues] = useState(initial);
  return (
    <div className="w-[360px]">
      <MultiSelector values={values} onValuesChange={setValues}>
        <MultiSelectorTrigger>
          <MultiSelectorInput aria-label="Fruit" placeholder="Pick fruit" />
        </MultiSelectorTrigger>
        <MultiSelectorContent>
          <MultiSelectorList>
            {FRUIT.map((option) => (
              <MultiSelectorItem
                key={option}
                value={option}
                disabled={option === disabled}
              >
                {option}
              </MultiSelectorItem>
            ))}
          </MultiSelectorList>
        </MultiSelectorContent>
      </MultiSelector>
    </div>
  );
}

export const Empty: Story = { render: () => <Picker initial={[]} /> };
export const WithValues: Story = {
  render: () => <Picker initial={["Apples", "Cherries"]} />,
};
export const WithDisabledOption: Story = {
  render: () => <Picker initial={["Apples"]} disabled="Dates" />,
};
