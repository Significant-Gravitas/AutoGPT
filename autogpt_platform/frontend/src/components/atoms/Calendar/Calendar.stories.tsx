import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { Calendar } from "./Calendar";

const meta: Meta<typeof Calendar> = {
  title: "Atoms/Calendar",
  component: Calendar,
  parameters: { layout: "centered" },
};

export default meta;
type Story = StoryObj<typeof Calendar>;

interface Props {
  initial?: Date;
}

function Single({ initial }: Props) {
  const [selected, setSelected] = useState<Date | undefined>(initial);
  return (
    <Calendar
      mode="single"
      selected={selected}
      onSelect={setSelected}
      defaultMonth={new Date(2026, 0, 1)}
    />
  );
}

export const Default: Story = { render: () => <Single /> };
export const WithSelection: Story = {
  render: () => <Single initial={new Date(2026, 0, 20)} />,
};
export const Range: Story = {
  render: () => (
    <Calendar
      mode="range"
      selected={{ from: new Date(2026, 0, 12), to: new Date(2026, 0, 16) }}
      defaultMonth={new Date(2026, 0, 1)}
    />
  ),
};
export const WithDisabledDays: Story = {
  render: () => (
    <Calendar
      mode="single"
      defaultMonth={new Date(2026, 0, 1)}
      disabled={{ dayOfWeek: [0, 6] }}
    />
  ),
};
