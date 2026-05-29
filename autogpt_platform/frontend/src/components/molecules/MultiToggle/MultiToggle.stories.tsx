import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { MultiToggle } from "./MultiToggle";

const meta: Meta<typeof MultiToggle> = {
  title: "Molecules/MultiToggle",
  tags: ["autodocs"],
  component: MultiToggle,
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "MultiToggle component that behaves like a checkbox group, allowing multiple items to be selected. Each item uses outline button styling with purple-600 accent for selected state.",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

const weekdayItems = [
  { value: "select-all", label: "Select all" },
  { value: "weekdays", label: "Weekdays" },
  { value: "weekends", label: "Weekends" },
];

const dayItems = [
  { value: "su", label: "Su" },
  { value: "mo", label: "Mo" },
  { value: "tu", label: "Tu" },
  { value: "we", label: "We" },
  { value: "th", label: "Th" },
  { value: "fr", label: "Fr" },
  { value: "sa", label: "Sa" },
];

export const WeekdaySelector: Story = {
  render: function WeekdaySelectorStory() {
    const [selectedValues, setSelectedValues] = useState<string[]>([]);

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">On</span>
          <MultiToggle
            items={weekdayItems}
            selectedValues={selectedValues}
            onChange={setSelectedValues}
            aria-label="Select scheduling options"
          />
        </div>
        <MultiToggle
          items={dayItems}
          selectedValues={selectedValues}
          onChange={setSelectedValues}
          aria-label="Select specific days"
        />
      </div>
    );
  },
};

export const SimpleExample: Story = {
  render: function SimpleExampleStory() {
    const [selectedValues, setSelectedValues] = useState<string[]>(["option2"]);

    return (
      <MultiToggle
        items={[
          { value: "option1", label: "Option 1" },
          { value: "option2", label: "Option 2" },
          { value: "option3", label: "Option 3" },
        ]}
        selectedValues={selectedValues}
        onChange={setSelectedValues}
        aria-label="Select options"
      />
    );
  },
};

export const WithDisabledItems: Story = {
  render: function WithDisabledItemsStory() {
    const [selectedValues, setSelectedValues] = useState<string[]>([
      "enabled1",
    ]);

    return (
      <MultiToggle
        items={[
          { value: "enabled1", label: "Enabled 1" },
          { value: "disabled1", label: "Disabled 1", disabled: true },
          { value: "enabled2", label: "Enabled 2" },
          { value: "disabled2", label: "Disabled 2", disabled: true },
        ]}
        selectedValues={selectedValues}
        onChange={setSelectedValues}
        aria-label="Select options with some disabled"
      />
    );
  },
};

export const AllSelected: Story = {
  render: function AllSelectedStory() {
    const [selectedValues, setSelectedValues] = useState<string[]>([
      "option1",
      "option2",
      "option3",
      "option4",
    ]);

    return (
      <MultiToggle
        items={[
          { value: "option1", label: "Option 1" },
          { value: "option2", label: "Option 2" },
          { value: "option3", label: "Option 3" },
          { value: "option4", label: "Option 4" },
        ]}
        selectedValues={selectedValues}
        onChange={setSelectedValues}
        aria-label="All options selected"
      />
    );
  },
};
