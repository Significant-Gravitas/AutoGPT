import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import type { OriginFilter as OriginFilterValue } from "../../useArtifactsPage";
import { OriginFilter } from "./OriginFilter";

const meta: Meta<typeof OriginFilter> = {
  title: "Pages/Artifacts/OriginFilter",
  component: OriginFilter,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "Segmented pill filter for the Artifacts page. The active pill " +
          "slides between options via Framer Motion's shared `layoutId`. " +
          "Drives the `origin` query param on `useListWorkspaceFiles` " +
          "(uploaded vs generated, all = no filter).",
      },
    },
  },
  argTypes: {
    value: {
      control: { type: "radio" },
      options: ["all", "uploaded", "generated"] satisfies OriginFilterValue[],
      description: "Currently selected origin filter.",
    },
    onChange: {
      action: "changed",
      description: "Fired when the user picks a different option.",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const All: Story = {
  args: { value: "all" },
};

export const Uploaded: Story = {
  args: { value: "uploaded" },
};

export const Generated: Story = {
  args: { value: "generated" },
};

export const Interactive: Story = {
  render: function InteractiveStory() {
    const [value, setValue] = useState<OriginFilterValue>("all");
    return (
      <div className="flex flex-col items-center gap-4">
        <OriginFilter value={value} onChange={setValue} />
        <span className="text-xs text-zinc-500">
          Selected: <code className="text-zinc-900">{value}</code>
        </span>
      </div>
    );
  },
  parameters: {
    docs: {
      description: {
        story:
          "Stateful playground — click pills to see the active background " +
          "slide between them.",
      },
    },
  },
};
