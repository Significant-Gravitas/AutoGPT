import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { FilterChip } from "./FilterChip";

const meta: Meta<typeof FilterChip> = {
  title: "Atoms/FilterChip",
  component: FilterChip,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
  },
  argTypes: {
    size: {
      control: "select",
      options: ["sm", "md", "lg"],
    },
  },
};

export default meta;
type Story = StoryObj<typeof FilterChip>;

export const Default: Story = {
  args: {
    label: "Marketing",
  },
};

export const Selected: Story = {
  args: {
    label: "Marketing",
    selected: true,
  },
};

export const Dismissible: Story = {
  args: {
    label: "Marketing",
    selected: true,
    dismissible: true,
  },
};

export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-4">
      <FilterChip label="Small" size="sm" />
      <FilterChip label="Medium" size="md" />
      <FilterChip label="Large" size="lg" />
    </div>
  ),
};

export const Disabled: Story = {
  args: {
    label: "Disabled",
    disabled: true,
  },
};

function FilterChipGroupDemo() {
  const filters = [
    "Marketing",
    "Sales",
    "Development",
    "Design",
    "Research",
    "Analytics",
  ];
  const [selected, setSelected] = useState<string[]>(["Marketing"]);

  function handleToggle(filter: string) {
    setSelected((prev) =>
      prev.includes(filter)
        ? prev.filter((f) => f !== filter)
        : [...prev, filter],
    );
  }

  return (
    <div className="flex flex-wrap gap-3">
      {filters.map((filter) => (
        <FilterChip
          key={filter}
          label={filter}
          selected={selected.includes(filter)}
          onClick={() => handleToggle(filter)}
        />
      ))}
    </div>
  );
}

export const FilterGroup: Story = {
  render: () => <FilterChipGroupDemo />,
};

function SingleSelectDemo() {
  const filters = ["All", "Featured", "Popular", "New"];
  const [selected, setSelected] = useState("All");

  return (
    <div className="flex flex-wrap gap-3">
      {filters.map((filter) => (
        <FilterChip
          key={filter}
          label={filter}
          selected={selected === filter}
          onClick={() => setSelected(filter)}
        />
      ))}
    </div>
  );
}

export const SingleSelect: Story = {
  render: () => <SingleSelectDemo />,
};

function DismissibleDemo() {
  const [filters, setFilters] = useState(["Marketing", "Sales", "Development"]);

  function handleDismiss(filter: string) {
    setFilters((prev) => prev.filter((f) => f !== filter));
  }

  return (
    <div className="flex flex-wrap gap-3">
      {filters.map((filter) => (
        <FilterChip
          key={filter}
          label={filter}
          selected
          dismissible
          onDismiss={() => handleDismiss(filter)}
        />
      ))}
      {filters.length === 0 && (
        <span className="text-neutral-500">No filters selected</span>
      )}
    </div>
  );
}

export const DismissibleGroup: Story = {
  render: () => <DismissibleDemo />,
};
