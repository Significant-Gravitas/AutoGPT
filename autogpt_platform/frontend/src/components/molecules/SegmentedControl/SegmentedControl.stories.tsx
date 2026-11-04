import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  CheckmarkCircle02Icon,
  Clock01Icon,
  InboxIcon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { SegmentedControl } from "./SegmentedControl";

const meta: Meta = {
  title: "Molecules/SegmentedControl",
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Switches between two to six options. A gray track with a white pill that glides to the picked option (shared layout animation, out-expo). Radiogroup semantics: arrow keys, Home and End move and select. Optional counts roll with NumberTicker.",
      },
    },
  },
};

export default meta;
type Story = StoryObj;

type Filter = "all" | "running" | "waiting" | "done";

export const WithCounts: Story = {
  render: function WithCountsStory() {
    const [value, setValue] = useState<Filter>("all");
    const [running, setRunning] = useState(3);
    return (
      <div className="flex flex-col items-center gap-4">
        <SegmentedControl<Filter>
          ariaLabel="Filter tasks"
          value={value}
          onChange={setValue}
          options={[
            { value: "all", label: "All", count: 12 + running },
            { value: "running", label: "Running", count: running },
            { value: "waiting", label: "Needs you", count: 2 },
            { value: "done", label: "Done", count: 10 },
          ]}
        />
        <Button
          variant="secondary"
          size="small"
          onClick={() => setRunning((n) => n + 1)}
        >
          Start one more
        </Button>
      </div>
    );
  },
};

export const WithIcons: Story = {
  render: function WithIconsStory() {
    const [value, setValue] = useState<Filter>("all");
    return (
      <SegmentedControl<Filter>
        ariaLabel="Filter tasks"
        value={value}
        onChange={setValue}
        options={[
          { value: "all", label: "All", icon: InboxIcon },
          { value: "running", label: "Running", icon: Clock01Icon },
          { value: "done", label: "Done", icon: CheckmarkCircle02Icon },
        ]}
      />
    );
  },
};

export const Small: Story = {
  render: function SmallStory() {
    const [value, setValue] = useState<"list" | "board">("list");
    return (
      <SegmentedControl<"list" | "board">
        ariaLabel="Layout"
        size="sm"
        value={value}
        onChange={setValue}
        options={[
          { value: "list", label: "List" },
          { value: "board", label: "Board" },
        ]}
      />
    );
  },
};
